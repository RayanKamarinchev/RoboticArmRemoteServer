let groups = [];
let instructions = [];
let editingGroupId = null;
let draggedElement = null;
let draggedFromAvailable = false;
let pendingInstruction = null;
let detectionPollInterval = null;

function convertCoordsMetric(coords, from_server){
    const convert = x => from_server
        ? Math.round(x * 1000)
        : x / 1000;

    if (Array.isArray(coords)) {
        return coords.map(convert);
    }

    return convert(coords);
}

loadData();
startDetectionPolling();

function startDetectionPolling() {
    // Poll for camera image and detected boxes
    detectionPollInterval = setInterval(() => {
        fetch('/api/get_cam_data')
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    updateCameraImage(data.image);
                    updateDetectedBoxes(data.boxes);
                }
            });
    }, 500);
}

function updateCameraImage(imageData) {
    const display = document.getElementById('cameraImage');
    if (imageData) {
        display.innerHTML = `<img src="data:image/jpeg;base64,${imageData}" alt="Camera feed">`;
    } else {
        display.innerHTML = '<div class="no-image">No image available</div>';
    }
}

function updateDetectedBoxes(boxes) {
    const list = document.getElementById('detectedBoxesList');
    if (!boxes || boxes.length === 0) {
        list.innerHTML = '<div class="empty-state">No boxes detected</div>';
        return;
    }

    list.innerHTML = boxes.map(box => {
        box.grab_point = convertCoordsMetric(box.grab_point, true);
        box.width = convertCoordsMetric(box.width, true);
        box.length = convertCoordsMetric(box.length, true);
        box.height = convertCoordsMetric(box.height, true);
        return `<div class="detected-box-item">
                    <div class="detected-box-id">Box ${box.id}</div>
                    <div class="detected-box-details">
                        <div>Position: (${box.grab_point[0].toFixed(1)}, ${box.grab_point[1].toFixed(1)}, ${box.grab_point[2].toFixed(1)}) mm</div>
                        <div>Size: ${box.width.toFixed(1)} × ${box.length.toFixed(1)} × ${box.height.toFixed(1)} mm</div>
                    </div>
                </div>
            `}).join('');
}

function loadData() {
    fetch('/api/data')
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                groups = data.data.groups || [];
                instructions = data.data.instructions || [];
                renderGroups();
                renderInstructions();
            }
        });
}

function renderGroups() {
    const list = document.getElementById('groupsList');
    if (groups.length === 0) {
        list.innerHTML = '<div class="empty-state">No groups created yet</div>';
        return;
    }

    list.innerHTML = groups.map(group => `
                <div class="group-card">
                    <div class="group-header">
                        <div class="group-name">${group.name}</div>
                        <div class="group-actions">
                            <button class="btn btn-small btn-primary" onclick="editGroup('${group.id}')">Edit</button>
                            <button class="btn btn-small btn-danger" onclick="deleteGroup('${group.id}')">Delete</button>
                        </div>
                    </div>
                    <div class="group-details">
                        <div class="group-location">Location: (${group.location.x}, ${group.location.y}, ${group.location.z})</div>
                        <div class="group-boxes">Boxes: ${group.boxes.length > 0 ? group.boxes.join(', ') : 'None'}</div>
                    </div>
                </div>
            `).join('');
}

function renderInstructions() {
    const list = document.getElementById('instructionList');

    list.innerHTML = '';

    if (instructions.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'empty-state';
        emptyDiv.textContent = 'Drag instructions here to build your program';
        emptyDiv.style.minHeight = '200px';

        // Make empty state droppable
        emptyDiv.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });

        emptyDiv.addEventListener('drop', (e) => {
            e.preventDefault();
            if (draggedFromAvailable) {
                const type = e.dataTransfer.getData('text/plain');
                const newInstruction = { type, params: {} };

                if (needsParameters(type)) {
                    pendingInstruction = { instruction: newInstruction, index: 0 };
                    openParamModal(type);
                } else {
                    instructions.splice(0, 0, newInstruction);
                    saveAndRender();
                }
            }
        });

        list.appendChild(emptyDiv);
        return;
    }

    instructions.forEach((instruction, index) => {
        list.appendChild(createInstructionElement(instruction, index));
    });
}

function createInstructionElement(instruction, index) {
    const div = document.createElement('div');
    div.className = 'instruction-item';
    div.draggable = true;
    div.dataset.index = index;

    const paramsText = getParamsText(instruction);

    div.innerHTML = `
                <div class="instruction-content">
                    <span class="drag-handle">☰</span>
                    <span class="instruction-type">${instruction.type}</span>
                    <span class="instruction-params">${paramsText}</span>
                </div>
                <div class="instruction-actions">
                    <button class="btn btn-small btn-primary" onclick="editInstruction(${index})">Edit</button>
                    <button class="btn btn-small btn-danger" onclick="deleteInstruction(${index})">Delete</button>
                </div>
            `;

    div.addEventListener('dragstart', handleDragStart);
    div.addEventListener('dragend', handleDragEnd);
    div.addEventListener('dragover', handleDragOver);
    div.addEventListener('drop', handleDrop);
    div.addEventListener('dragleave', handleDragLeave);

    return div;
}

function getParamsText(instruction) {
    const params = instruction.params || {};
    switch (instruction.type) {
        case 'wait':
            return params.time ? `${params.time} ms` : 'No time set';
        case 'go_box':
            return params.box_id >= 0 ? `Box: ${params.box_id}` : 'No box set';
        case 'go_group_box':
            return params.group_id ? `Group: ${params.group_id}` : 'No group set';
        case 'go_pos':
            return params.x !== undefined ? `(${params.x}, ${params.y}, ${params.z})` : 'No position set';
        case 'go_group_location':
            return params.group_id ? `Group: ${params.group_id}` : 'No group set';
        default:
            return '';
    }
}

// Drag and drop for available instructions
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('availableInstructions').addEventListener('dragstart', (e) => {
        if (e.target.classList.contains('available-instruction')) {
            draggedFromAvailable = true;
            draggedElement = e.target;
            e.dataTransfer.effectAllowed = 'copy';
            e.dataTransfer.setData('text/plain', e.target.dataset.type);
            e.target.style.opacity = '0.5';
        }
    });

    document.getElementById('availableInstructions').addEventListener('dragend', (e) => {
        if (e.target.classList.contains('available-instruction')) {
            e.target.style.opacity = '1';
        }
    });
});

function handleDragStart(e) {
    draggedElement = this;
    draggedFromAvailable = false;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', this.dataset.index);
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    draggedFromAvailable = false;
    document.querySelectorAll('.instruction-item').forEach(item => {
        item.classList.remove('drag-over');
    });
}

function handleDragOver(e) {
    e.preventDefault();

    if (!this.classList.contains('dragging')) {
        this.classList.add('drag-over');
    }

    e.dataTransfer.dropEffect = draggedFromAvailable ? 'copy' : 'move';
    return false;
}

function handleDragLeave(e) {
    this.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();

    this.classList.remove('drag-over');

    const dropIndex = parseInt(this.dataset.index);

    if (draggedFromAvailable) {
        const type = e.dataTransfer.getData('text/plain');
        const newInstruction = { type, params: {} };

        // Check if instruction needs parameters
        if (needsParameters(type)) {
            pendingInstruction = { instruction: newInstruction, index: dropIndex };
            openParamModal(type);
        } else {
            instructions.splice(dropIndex, 0, newInstruction);
            saveAndRender();
        }
    } else {
        const dragIndex = parseInt(draggedElement.dataset.index);
        if (dragIndex !== dropIndex && !isNaN(dragIndex)) {
            const item = instructions.splice(dragIndex, 1)[0];
            const newIndex = dragIndex < dropIndex ? dropIndex - 1 : dropIndex;
            instructions.splice(newIndex, 0, item);
            saveAndRender();
        }
    }

    draggedFromAvailable = false;
    return false;
}

function needsParameters(type) {
    return ['wait', 'go_box', 'go_group_box', 'go_pos', 'go_group_location'].includes(type);
}

function deleteInstruction(index) {
    instructions.splice(index, 1);
    saveAndRender();
}

function editInstruction(index) {
    const instruction = instructions[index];
    pendingInstruction = { instruction, index, editing: true };
    openParamModal(instruction.type, instruction.params);
}

function openParamModal(type, existingParams = {}) {
    const modal = document.getElementById('paramModal');
    const title = document.getElementById('paramModalTitle');
    const form = document.getElementById('paramForm');

    title.textContent = `Set Parameters for ${type}`;
    form.innerHTML = '';

    switch (type) {
        case 'wait':
            form.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Wait Time (ms)</label>
                            <input type="number" class="form-input" id="paramTime" value="${existingParams.time || ''}" placeholder="Enter time in milliseconds">
                        </div>
                    `;
            break;
        case 'go_box':
            form.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Box ID</label>
                            <input type="text" class="form-input" id="paramBoxId" value="${existingParams.box_id || ''}" placeholder="Enter box ID">
                        </div>
                    `;
            break;
        case 'go_group_box':
        case 'go_group_location':
            const groupOptions = groups.map(g => `<option value="${g.id}" ${existingParams.group_id === g.id ? 'selected' : ''}>${g.name}</option>`).join('');
            form.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Select Group</label>
                            <select class="form-input" id="paramGroupId">
                                <option value="">Select a group</option>
                                ${groupOptions}
                            </select>
                        </div>
                    `;
            break;
        case 'go_pos':
            form.innerHTML = `
                        <div class="form-group">
                            <label class="form-label">Position (x, y, z) in cm</label>
                            <div class="coord-inputs">
                                <input type="number" class="form-input" id="paramX" value="${existingParams.x || ''}" placeholder="X" step="0.1">
                                <input type="number" class="form-input" id="paramY" value="${existingParams.y || ''}" placeholder="Y" step="0.1">
                                <input type="number" class="form-input" id="paramZ" value="${existingParams.z || ''}" placeholder="Z" step="0.1">
                            </div>
                        </div>
                    `;
            break;
    }

    modal.classList.add('show');
}

function closeParamModal() {
    document.getElementById('paramModal').classList.remove('show');
    pendingInstruction = null;
}

function saveInstructionParams() {
    if (!pendingInstruction) return;

    const { instruction, index, editing } = pendingInstruction;
    const type = instruction.type;

    switch (type) {
        case 'wait':
            instruction.params = { time: parseInt(document.getElementById('paramTime').value) };
            break;
        case 'go_box':
            instruction.params = { box_id: parseInt(document.getElementById('paramBoxId').value) };
            break;
        case 'go_group_box':
        case 'go_group_location':
            instruction.params = { group_id: document.getElementById('paramGroupId').value };
            break;
        case 'go_pos':
            instruction.params = {
                x: parseFloat(document.getElementById('paramX').value),
                y: parseFloat(document.getElementById('paramY').value),
                z: parseFloat(document.getElementById('paramZ').value)
            };
            break;
    }

    if (editing) {
        instructions[index] = instruction;
    } else {
        instructions.splice(index, 0, instruction);
    }

    closeParamModal();
    saveAndRender();
}

function saveAndRender() {
    fetch('/api/instructions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instructions: instructions })
    }).then(res => res.json()).then(data => {
        if (data.success) {
            renderInstructions();
        }
    });
}

// Group management
function openGroupModal(groupId = null) {
    const modal = document.getElementById('groupModal');
    const title = document.getElementById('groupModalTitle');

    editingGroupId = groupId;

    if (groupId) {
        const group = groups.find(g => g.id === groupId);
        title.textContent = 'Edit Group';
        document.getElementById('groupName').value = group.name;
        document.getElementById('groupX').value = group.location.x;
        document.getElementById('groupY').value = group.location.y;
        document.getElementById('groupZ').value = group.location.z;
        renderGroupBoxTags(group.boxes);
    } else {
        title.textContent = 'Add Group';
        document.getElementById('groupName').value = '';
        document.getElementById('groupX').value = '';
        document.getElementById('groupY').value = '';
        document.getElementById('groupZ').value = '';
        renderGroupBoxTags([]);
    }

    modal.classList.add('show');
}

function closeGroupModal() {
    document.getElementById('groupModal').classList.remove('show');
    editingGroupId = null;
}

let currentBoxes = [];

function renderGroupBoxTags(boxes = currentBoxes) {
    currentBoxes = boxes;
    const container = document.getElementById('groupBoxTags');
    container.innerHTML = boxes.map(box => `
                <div class="box-tag">
                    ${box}
                    <span class="box-tag-remove" onclick="removeBox(${box})">×</span>
                </div>
            `).join('');
}

document.getElementById('groupBoxInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const input = e.target;
        const boxId = input.value.trim();
        if (boxId && !currentBoxes.includes(boxId)) {
            currentBoxes.push(boxId);
            renderGroupBoxTags();
            input.value = '';
        }
    }
});

function removeBox(boxId) {
    currentBoxes = currentBoxes.filter(b => b !== boxId);
    console.log(currentBoxes);
    renderGroupBoxTags();
}

function saveGroup() {
    const name = document.getElementById('groupName').value.trim();
    const x = parseFloat(document.getElementById('groupX').value);
    const y = parseFloat(document.getElementById('groupY').value);
    const z = parseFloat(document.getElementById('groupZ').value);

    if (!name) {
        alert('Please enter a group name');
        return;
    }

    const groupData = {
        name,
        location: { x, y, z },
        boxes: currentBoxes.map(parseInt)
    };

    if (editingGroupId) {
        groupData.id = editingGroupId;
        fetch(`/api/groups/${editingGroupId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(groupData)
        }).then(res => res.json()).then(data => {
            if (data.success) {
                loadData();
                closeGroupModal();
            }
        });
    } else {
        fetch('/api/groups', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(groupData)
        }).then(res => res.json()).then(data => {
            if (data.success) {
                loadData();
                closeGroupModal();
            }
        });
    }
}

function editGroup(groupId) {
    openGroupModal(groupId);
}

function deleteGroup(groupId) {
    if (confirm('Are you sure you want to delete this group?')) {
        fetch(`/api/groups/${groupId}`, {
            method: 'DELETE'
        }).then(res => res.json()).then(data => {
            if (data.success) {
                loadData();
            }
        });
    }
}