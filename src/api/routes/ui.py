"""Simple web UI for testing weed detection."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
@router.get("/visualize", response_class=HTMLResponse)
async def detection_ui():
    """Serve a simple web UI for testing weed detection with visualization."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weed Detection - Visualizer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4aa; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 30px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab {
            padding: 12px 24px; border: none; border-radius: 8px 8px 0 0;
            background: #16213e; color: #888; cursor: pointer; font-size: 16px;
        }
        .tab.active { background: #00d4aa; color: #000; font-weight: 600; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .upload-section {
            background: #16213e;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #00d4aa;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover { background: rgba(0, 212, 170, 0.1); }
        .upload-area.dragover { background: rgba(0, 212, 170, 0.2); border-color: #00ff88; }
        input[type="file"] { display: none; }
        .options { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
        .option { display: flex; align-items: center; gap: 8px; }
        label { color: #aaa; }
        input[type="number"] {
            width: 80px; padding: 8px; border: 1px solid #333;
            border-radius: 4px; background: #0f0f23; color: #fff;
        }
        select {
            padding: 8px 12px; border: 1px solid #333;
            border-radius: 4px; background: #0f0f23; color: #fff;
            cursor: pointer;
        }
        select:hover { border-color: #00d4aa; }
        .model-description { font-size: 12px; color: #666; margin-top: 5px; }
        button {
            background: #00d4aa; color: #000; border: none;
            padding: 12px 30px; border-radius: 6px;
            font-size: 16px; font-weight: 600; cursor: pointer;
        }
        button:hover { background: #00ff88; }
        button:disabled { background: #444; cursor: not-allowed; }
        .results { display: none; }
        .results.show { display: block; }
        .images-container {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 20px; margin-bottom: 30px;
        }
        @media (max-width: 900px) { .images-container { grid-template-columns: 1fr; } }
        .image-box { background: #16213e; border-radius: 12px; overflow: hidden; }
        .image-box h3 {
            margin: 0; padding: 15px 20px; background: #0f0f23;
            color: #00d4aa; font-size: 14px; text-transform: uppercase;
        }
        .image-box img { width: 100%; display: block; }
        .stats { background: #16213e; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        .stats h3 { margin-top: 0; color: #00d4aa; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }
        .stat { background: #0f0f23; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 28px; font-weight: bold; color: #00d4aa; }
        .stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
        .detections-list { background: #16213e; padding: 20px; border-radius: 12px; }
        .detections-list h3 { margin-top: 0; color: #00d4aa; }
        .detection-item {
            display: flex; align-items: center; gap: 15px;
            padding: 12px; background: #0f0f23; border-radius: 6px; margin-bottom: 10px;
        }
        .detection-label { font-weight: 600; min-width: 100px; }
        .confidence-bar { flex: 1; height: 20px; background: #333; border-radius: 10px; overflow: hidden; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #00d4aa, #00ff88); border-radius: 10px; }
        .confidence-value { min-width: 50px; text-align: right; font-weight: bold; }
        .loading { display: none; text-align: center; padding: 40px; }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #333; border-top: 4px solid #00d4aa;
            border-radius: 50%; width: 50px; height: 50px;
            animation: spin 1s linear infinite; margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #ff4444; color: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: none; }
        .error.show { display: block; }
        .comparison-section { margin-top: 30px; }
        .comparison-section img { width: 100%; border-radius: 12px; }
        .weed-colors { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 10px; }
        .weed-color { display: flex; align-items: center; gap: 8px; font-size: 14px; }
        .color-dot { width: 16px; height: 16px; border-radius: 50%; }
        .ref-section { background: #16213e; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        .ref-section h3 { margin-top: 0; color: #00d4aa; }
        .ref-form { display: flex; gap: 15px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 20px; }
        .ref-form .field { display: flex; flex-direction: column; gap: 5px; }
        .ref-form input[type="text"] {
            padding: 10px; border: 1px solid #333; border-radius: 4px;
            background: #0f0f23; color: #fff; min-width: 150px;
        }
        .ref-types { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }
        .ref-type-badge {
            display: flex; align-items: center; gap: 8px;
            background: #0f0f23; padding: 8px 15px; border-radius: 20px;
        }
        .ref-type-badge .name { font-weight: 600; text-transform: capitalize; }
        .ref-type-badge .count { color: #00d4aa; font-size: 14px; }
        .success-msg { color: #00d4aa; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Weed Detection</h1>
    <p class="subtitle">AI-powered weed detection with visualization</p>

    <div class="tabs">
        <button class="tab active" onclick="switchTab('detect')">Detect Weeds</button>
        <button class="tab" onclick="switchTab('references')">Manage References</button>
    </div>

    <div id="detect-tab" class="tab-content active">
    <div class="upload-section">
        <div class="upload-area" id="uploadArea">
            <p>Drop an image here or click to upload</p>
            <p style="color: #666; font-size: 14px;">Supports JPG, PNG</p>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        <div class="options">
            <div class="option">
                <label for="detectionMode">Detection Mode:</label>
                <select id="detectionMode" onchange="updateModeDescription()">
                    <option value="text_owlv2" selected>OWLv2 Text Detection</option>
                    <optgroup label="DINO Models">
                        <option value="grounding_dino">DINO Tiny (Default)</option>
                        <option value="grounding_dino_1_5_edge">DINO Tiny - Fast ⚡</option>
                        <option value="grounding_dino_1_5_pro">DINO Base - Accurate 🎯</option>
                        <option value="dynamic_dino">DINO Tiny - Balanced</option>
                    </optgroup>
                    <optgroup label="Local Weights 📦">
                        <option value="grounding_dino_local_swint">DINO Swin-T (Local) - Fast</option>
                        <option value="grounding_dino_local_swinb">DINO Swin-B (Local) - Accurate</option>
                    </optgroup>
                    <optgroup label="Fine-Tuned Models 🎯">
                        <option value="rf_detr">RF-DETR (Weed Detection)</option>
                    </optgroup>
                    <option value="sam_auto">SAM Auto-Segment</option>
                    <option value="image_owlv2">OWLv2 Image Detection</option>
                </select>
            </div>
            <div class="option" id="tensorrtOption" style="display: none;">
                <label for="useTensorrt">
                    <input type="checkbox" id="useTensorrt" style="display: inline; width: auto; margin-right: 5px;">
                    TensorRT Acceleration
                </label>
                <span style="font-size: 11px; color: #666; margin-left: 5px;">(NVIDIA GPU)</span>
            </div>
            <div class="option">
                <label for="threshold">Confidence Threshold:</label>
                <input type="number" id="threshold" value="0.7" min="0" max="1" step="0.05">
            </div>
            <div class="option">
                <label for="groupOverlapping">
                    <input type="checkbox" id="groupOverlapping" style="display: inline; width: auto; margin-right: 5px;">
                    Group overlapping
                </label>
            </div>
        </div>
        <div id="weedTypeSelector" class="weed-type-selector" style="display: none; margin-bottom: 15px; padding: 15px; background: #0f0f23; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <label style="color: #00d4aa; font-weight: 600;">Detect Weed Types:</label>
                <div style="display: flex; gap: 10px;">
                    <button type="button" onclick="selectAllWeeds()" style="padding: 4px 10px; font-size: 12px; background: #1a1a2e; border: 1px solid #00d4aa; color: #00d4aa; border-radius: 4px; cursor: pointer;">Select All</button>
                    <button type="button" onclick="unselectAllWeeds()" style="padding: 4px 10px; font-size: 12px; background: #1a1a2e; border: 1px solid #888; color: #888; border-radius: 4px; cursor: pointer;">Unselect All</button>
                </div>
            </div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedDandelion" checked style="width: auto;">
                    <span class="color-dot" style="background: #FFD700; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Dandelion
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedClover" checked style="width: auto;">
                    <span class="color-dot" style="background: #32CD32; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Clover
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedCrabgrass" checked style="width: auto;">
                    <span class="color-dot" style="background: #FF6347; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Crabgrass
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedPoaAnnua" checked style="width: auto;">
                    <span class="color-dot" style="background: #87CEEB; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Poa Annua
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedSilverleaf" style="width: auto;">
                    <span class="color-dot" style="background: #9370DB; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Silverleaf Nightshade
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedBindweed" style="width: auto;">
                    <span class="color-dot" style="background: #FF69B4; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Field Bindweed
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedSnakeweed" style="width: auto;">
                    <span class="color-dot" style="background: #DAA520; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Broom Snakeweed
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedAmaranth" style="width: auto;">
                    <span class="color-dot" style="background: #DC143C; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Palmer's Amaranth
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                    <input type="checkbox" id="weedRussianThistle" style="width: auto;">
                    <span class="color-dot" style="background: #8B4513; width: 12px; height: 12px; border-radius: 50%;"></span>
                    Russian Thistle
                </label>
            </div>
        </div>
        <p id="modeDescription" class="model-description">Uses text descriptions to find objects. Best for flowers and distinctive features.</p>
        <button id="detectBtn" disabled>Detect Weeds</button>
        <div class="weed-colors">
            <span style="color: #888;">Legend:</span>
            <div class="weed-color"><div class="color-dot" style="background: #FFD700;"></div> Dandelion</div>
            <div class="weed-color"><div class="color-dot" style="background: #32CD32;"></div> Clover</div>
            <div class="weed-color"><div class="color-dot" style="background: #FF6347;"></div> Crabgrass</div>
            <div class="weed-color"><div class="color-dot" style="background: #87CEEB;"></div> Poa Annua</div>
            <div class="weed-color"><div class="color-dot" style="background: #9370DB;"></div> Silverleaf</div>
            <div class="weed-color"><div class="color-dot" style="background: #FF69B4;"></div> Bindweed</div>
            <div class="weed-color"><div class="color-dot" style="background: #DAA520;"></div> Snakeweed</div>
            <div class="weed-color"><div class="color-dot" style="background: #DC143C;"></div> Amaranth</div>
            <div class="weed-color"><div class="color-dot" style="background: #8B4513;"></div> R. Thistle</div>
        </div>
    </div>

    <div class="error" id="error"></div>
    <div class="loading" id="loading"><div class="spinner"></div><p>Analyzing image...</p></div>

    <div class="results" id="results">
        <div class="stats">
            <h3>Detection Summary</h3>
            <div class="stat-grid">
                <div class="stat"><div class="stat-value" id="weedCount">0</div><div class="stat-label">Weeds Found</div></div>
                <div class="stat"><div class="stat-value" id="inferenceTime">0</div><div class="stat-label">Inference (ms)</div></div>
                <div class="stat"><div class="stat-value" id="imageSize">-</div><div class="stat-label">Image Size</div></div>
            </div>
        </div>
        <div class="images-container">
            <div class="image-box"><h3>Original Image</h3><img id="originalImg" src="" alt="Original"></div>
            <div class="image-box"><h3>Detection Result</h3><img id="annotatedImg" src="" alt="Annotated"></div>
        </div>
        <div class="detections-list"><h3>Detected Weeds</h3><div id="detectionsContainer"></div></div>
        <div class="comparison-section">
            <h3 style="color: #00d4aa;">Side-by-Side Comparison</h3>
            <img id="comparisonImg" src="" alt="Comparison">
        </div>
    </div>
    </div><!-- end detect-tab -->

    <div id="references-tab" class="tab-content">
        <div class="ref-section">
            <h3>Upload Reference Images</h3>
            <p style="color: #888; margin-bottom: 15px;">Reference images teach the detector what to look for. Upload 5-10 diverse images per weed type.</p>
            <p style="color: #00d4aa; margin-bottom: 15px; font-size: 13px;"><strong>Tip:</strong> For best results, use tightly cropped images showing <em>only</em> the weed with minimal background. Too much grass in reference images causes over-detection.</p>
            <div class="ref-form">
                <div class="field">
                    <label for="weedType">Weed Type</label>
                    <input type="text" id="weedType" placeholder="e.g., dandelion">
                </div>
                <div class="field">
                    <label for="refFileInput">Image</label>
                    <input type="file" id="refFileInput" accept="image/*" style="display: block; padding: 8px;">
                </div>
                <button id="uploadRefBtn" disabled>Upload Reference</button>
            </div>
            <div id="refSuccess" class="success-msg" style="display: none;"></div>
            <div id="refError" class="error" style="display: none;"></div>
        </div>
        <div class="ref-section">
            <h3>Available Weed Types</h3>
            <p style="color: #888; margin-bottom: 10px;">These are the weed types you can detect:</p>
            <div id="refTypesContainer" class="ref-types">
                <p style="color: #666;">Loading...</p>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const detectBtn = document.getElementById('detectBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const errorDiv = document.getElementById('error');
        let selectedFile = null;

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) { showError('Please select an image file'); return; }
            selectedFile = file;
            uploadArea.textContent = 'Selected: ' + file.name;
            detectBtn.disabled = false;
        }

        detectBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            const threshold = document.getElementById('threshold').value;
            const detectionMode = document.getElementById('detectionMode').value;
            const groupOverlapping = document.getElementById('groupOverlapping').checked;
            const useTensorrt = document.getElementById('useTensorrt').checked;
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('confidence_threshold', threshold);
            formData.append('detection_mode', detectionMode);
            formData.append('group_overlapping', groupOverlapping);
            formData.append('use_tensorrt', useTensorrt);

            // For DINO modes and RF-DETR, include selected weed types
            const needsWeedTypes = detectionMode.includes('dino') || detectionMode === 'rf_detr';
            if (needsWeedTypes) {
                const selectedTypes = [];
                if (document.getElementById('weedDandelion').checked) selectedTypes.push('dandelion');
                if (document.getElementById('weedClover').checked) selectedTypes.push('clover');
                if (document.getElementById('weedCrabgrass').checked) selectedTypes.push('crabgrass');
                if (document.getElementById('weedPoaAnnua').checked) selectedTypes.push('poa_annua');
                if (document.getElementById('weedSilverleaf').checked) selectedTypes.push('silverleaf_nightshade');
                if (document.getElementById('weedBindweed').checked) selectedTypes.push('field_bindweed');
                if (document.getElementById('weedSnakeweed').checked) selectedTypes.push('broom_snakeweed');
                if (document.getElementById('weedAmaranth').checked) selectedTypes.push('palmers_amaranth');
                if (document.getElementById('weedRussianThistle').checked) selectedTypes.push('russian_thistle');
                if (selectedTypes.length > 0) {
                    formData.append('weed_types', selectedTypes.join(','));
                }
            }

            loading.classList.add('show');
            results.classList.remove('show');
            errorDiv.classList.remove('show');
            detectBtn.disabled = true;

            try {
                const response = await fetch('/detect/visualize', { method: 'POST', body: formData });
                const data = await response.json();
                if (!data.success) throw new Error(data.error || 'Detection failed');
                displayResults(data);
            } catch (err) {
                showError(err.message);
            } finally {
                loading.classList.remove('show');
                detectBtn.disabled = false;
            }
        });

        function displayResults(data) {
            const result = data.result;
            document.getElementById('weedCount').textContent = result.detections.length;
            document.getElementById('inferenceTime').textContent = result.inference_time_ms.toFixed(0);
            document.getElementById('imageSize').textContent = result.image_width + ' x ' + result.image_height;
            document.getElementById('originalImg').src = 'data:image/jpeg;base64,' + data.original_image;
            document.getElementById('annotatedImg').src = 'data:image/jpeg;base64,' + data.annotated_image;
            document.getElementById('comparisonImg').src = 'data:image/jpeg;base64,' + data.comparison_image;

            const container = document.getElementById('detectionsContainer');
            container.textContent = '';
            if (result.detections.length === 0) {
                container.textContent = 'No weeds detected above threshold';
            } else {
                result.detections.forEach(d => {
                    const item = document.createElement('div');
                    item.className = 'detection-item';
                    const label = document.createElement('span');
                    label.className = 'detection-label';
                    label.textContent = d.label;
                    const barContainer = document.createElement('div');
                    barContainer.className = 'confidence-bar';
                    const fill = document.createElement('div');
                    fill.className = 'confidence-fill';
                    fill.style.width = (d.confidence * 100) + '%';
                    barContainer.appendChild(fill);
                    const value = document.createElement('span');
                    value.className = 'confidence-value';
                    value.textContent = (d.confidence * 100).toFixed(1) + '%';
                    item.appendChild(label);
                    item.appendChild(barContainer);
                    item.appendChild(value);
                    container.appendChild(item);
                });
            }
            results.classList.add('show');
        }

        function showError(message) {
            errorDiv.textContent = 'Error: ' + message;
            errorDiv.classList.add('show');
        }

        function updateModeDescription() {
            const mode = document.getElementById('detectionMode').value;
            const desc = document.getElementById('modeDescription');
            const weedSelector = document.getElementById('weedTypeSelector');
            const tensorrtOption = document.getElementById('tensorrtOption');
            const descriptions = {
                'text_owlv2': 'Uses text descriptions to find objects. Best for flowers and distinctive features.',
                'grounding_dino': 'Grounding DINO Tiny (1024px images). Good balance of speed and accuracy.',
                'grounding_dino_local_swint': 'Local Swin-T weights (~8 FPS). Requires groundingdino_swint_ogc.pth in weights/ folder.',
                'grounding_dino_local_swinb': 'Local Swin-B weights (~5 FPS, more accurate). Requires groundingdino_swinb_cogcoor.pth in weights/ folder.',
                'grounding_dino_1_5_edge': 'Grounding DINO Tiny with smaller images (640px). Faster inference, slightly less detail.',
                'grounding_dino_1_5_pro': 'Grounding DINO Base with larger images (1024px). Best accuracy, uses more memory.',
                'dynamic_dino': 'Grounding DINO Tiny with balanced settings (800px). Middle ground option.',
                'rf_detr': 'RF-DETR fine-tuned on weed dataset. Highest accuracy for whole-plant detection. ~30 FPS. Requires trained weights.',
                'sam_auto': 'Finds all object boundaries automatically. Discovers plant regions without text prompts. Slower but thorough.',
                'image_owlv2': 'Uses reference images to find similar objects. Good when you have example images uploaded.'
            };
            desc.textContent = descriptions[mode] || '';
            // Show weed type selector for DINO modes and RF-DETR
            const showWeedSelector = mode.includes('dino') || mode === 'rf_detr';
            weedSelector.style.display = showWeedSelector ? 'block' : 'none';
            // Show TensorRT option for DINO 1.5 Edge, Pro, and Dynamic-DINO
            const supportsTensorrt = ['grounding_dino_1_5_edge', 'grounding_dino_1_5_pro', 'dynamic_dino'].includes(mode);
            tensorrtOption.style.display = supportsTensorrt ? 'flex' : 'none';
        }

        function selectAllWeeds() {
            document.getElementById('weedDandelion').checked = true;
            document.getElementById('weedClover').checked = true;
            document.getElementById('weedCrabgrass').checked = true;
            document.getElementById('weedPoaAnnua').checked = true;
            document.getElementById('weedSilverleaf').checked = true;
            document.getElementById('weedBindweed').checked = true;
            document.getElementById('weedSnakeweed').checked = true;
            document.getElementById('weedAmaranth').checked = true;
            document.getElementById('weedRussianThistle').checked = true;
        }

        function unselectAllWeeds() {
            document.getElementById('weedDandelion').checked = false;
            document.getElementById('weedClover').checked = false;
            document.getElementById('weedCrabgrass').checked = false;
            document.getElementById('weedPoaAnnua').checked = false;
            document.getElementById('weedSilverleaf').checked = false;
            document.getElementById('weedBindweed').checked = false;
            document.getElementById('weedSnakeweed').checked = false;
            document.getElementById('weedAmaranth').checked = false;
            document.getElementById('weedRussianThistle').checked = false;
        }

        // Tab switching
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
            if (tabName === 'references') loadWeedTypes();
        }

        // Reference image upload
        const weedTypeInput = document.getElementById('weedType');
        const refFileInput = document.getElementById('refFileInput');
        const uploadRefBtn = document.getElementById('uploadRefBtn');
        const refSuccess = document.getElementById('refSuccess');
        const refErrorDiv = document.getElementById('refError');

        function checkRefForm() {
            uploadRefBtn.disabled = !weedTypeInput.value.trim() || !refFileInput.files.length;
        }
        weedTypeInput.addEventListener('input', checkRefForm);
        refFileInput.addEventListener('change', checkRefForm);

        uploadRefBtn.addEventListener('click', async () => {
            const weedType = weedTypeInput.value.trim();
            const file = refFileInput.files[0];
            if (!weedType || !file) return;

            refSuccess.style.display = 'none';
            refErrorDiv.style.display = 'none';
            uploadRefBtn.disabled = true;
            uploadRefBtn.textContent = 'Uploading...';

            const formData = new FormData();
            formData.append('image', file);
            formData.append('weed_type', weedType);

            try {
                const response = await fetch('/references/upload', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.success) {
                    refSuccess.textContent = data.message;
                    refSuccess.style.display = 'block';
                    refFileInput.value = '';
                    loadWeedTypes();
                } else {
                    throw new Error(data.detail || 'Upload failed');
                }
            } catch (err) {
                refErrorDiv.textContent = 'Error: ' + err.message;
                refErrorDiv.style.display = 'block';
            } finally {
                uploadRefBtn.disabled = false;
                uploadRefBtn.textContent = 'Upload Reference';
                checkRefForm();
            }
        });

        async function loadWeedTypes() {
            const container = document.getElementById('refTypesContainer');
            container.textContent = '';
            try {
                const response = await fetch('/references/weed-types');
                const data = await response.json();
                if (data.weed_types.length === 0) {
                    const p = document.createElement('p');
                    p.style.color = '#888';
                    p.textContent = 'No reference images yet. Upload some above!';
                    container.appendChild(p);
                } else {
                    data.weed_types.forEach(t => {
                        const badge = document.createElement('div');
                        badge.className = 'ref-type-badge';
                        const name = document.createElement('span');
                        name.className = 'name';
                        name.textContent = t.name;
                        const count = document.createElement('span');
                        count.className = 'count';
                        count.textContent = t.reference_count + ' images';
                        badge.appendChild(name);
                        badge.appendChild(count);
                        container.appendChild(badge);
                    });
                }
            } catch (err) {
                const p = document.createElement('p');
                p.style.color = '#ff4444';
                p.textContent = 'Failed to load weed types';
                container.appendChild(p);
            }
        }

        // Load weed types on page load
        loadWeedTypes();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)
