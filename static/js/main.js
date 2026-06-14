/* main.js — lógica de upload, preview e exibição dos resultados */
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultSection = document.getElementById('result-section');
    const loader = document.getElementById('loader');
    const previewImg = document.getElementById('preview-img');
    const categoryMain = document.getElementById('category-main');
    const confidenceFill = document.querySelector('.confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const topkList = document.getElementById('topk-list');
    const attrsList = document.getElementById('attrs-list');
    const jsonOutput = document.getElementById('json-output');
    const btnCopy = document.getElementById('btn-copy');
    const modelSelect = document.getElementById('model-select');
    const noTtaCheck = document.getElementById('no-tta');

    let lastFile = null;

    // Drag & drop visuals
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    });
    dropZone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });
    fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Re-executa predição automaticamente quando o modelo ou TTA mudar
    modelSelect.addEventListener('change', () => {
        if (lastFile) handleFile(lastFile);
    });
    noTtaCheck.addEventListener('change', () => {
        if (lastFile) handleFile(lastFile);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Por favor envie um ficheiro de imagem.');
            return;
        }
        lastFile = file;
        // Preview local
        const url = URL.createObjectURL(file);
        previewImg.src = url;

        // UI states
        resultSection.style.display = 'none';
        loader.style.display = 'block';

        const noTta = noTtaCheck.checked;
        const model = modelSelect.value;
        const loaderText = document.getElementById('loader-text');
        const modelLabel = modelSelect.options[modelSelect.selectedIndex].text;
        loaderText.textContent = `A analisar com ${modelLabel}…`;
        const formData = new FormData();
        formData.append('image', file);
        formData.append('no_tta', noTta ? 'true' : 'false');
        formData.append('model', model);

        fetch('/predict', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                loader.style.display = 'none';
                if (data.error) {
                    alert('Erro: ' + data.error);
                    return;
                }
                showResults(data);
            })
            .catch(err => {
                loader.style.display = 'none';
                alert('Erro de comunicação com o servidor.');
                console.error(err);
            });
    }

    function showResults(data) {
        resultSection.style.display = 'block';

        // Categoria principal
        categoryMain.textContent = data.category.label;
        const pct = Math.round(data.category.confidence * 100);
        confidenceFill.style.setProperty('--w', pct + '%');
        // Update pseudo-element width via inline style on the element itself is tricky,
        // we use a CSS custom property or direct style on a child. Let's use inline width on the ::after via a child div if needed.
        // Actually the simplest: replace the innerHTML of confidence-fill with a div representing the bar
        confidenceFill.innerHTML = `<div style="position:absolute;left:0;top:0;bottom:0;width:${pct}%;background:linear-gradient(90deg,#7c3aed,#3b82f6);border-radius:4px;"></div>`;
        confidenceText.textContent = pct + '%';

        // Top-K
        topkList.innerHTML = '';
        data.category.top_k.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<span>${item.rank}. ${item.label}</span><span class="score">${Math.round(item.confidence * 100)}%</span>`;
            topkList.appendChild(li);
        });

        // Cores
        const colorsList = document.getElementById('colors-list');
        const colorSwatch = document.getElementById('color-swatch');
        colorsList.innerHTML = '';
        if (data.colors && data.colors.length > 0) {
            const topColor = data.colors[0].name;
            colorSwatch.innerHTML = `<span class="color-name">${topColor}</span>`;
            colorSwatch.style.borderLeftColor = topColor;
            data.colors.forEach(c => {
                const span = document.createElement('span');
                span.className = 'tag color-tag';
                span.textContent = c.name;
                span.title = `confiança: ${Math.round(c.confidence * 100)}%`;
                colorsList.appendChild(span);
            });
        } else {
            colorSwatch.innerHTML = '<span class="color-name">—</span>';
        }

        // Atributos
        attrsList.innerHTML = '';
        if (data.attributes.length === 0) {
            attrsList.innerHTML = '<span class="tag" style="opacity:0.6;">Nenhum atributo acima do threshold</span>';
        } else {
            data.attributes.forEach(attr => {
                const span = document.createElement('span');
                span.className = 'tag';
                span.textContent = attr.name;
                span.title = `confiança: ${Math.round(attr.confidence * 100)}%`;
                attrsList.appendChild(span);
            });
        }

        // JSON
        jsonOutput.textContent = JSON.stringify(data, null, 2);
    }

    btnCopy.addEventListener('click', () => {
        navigator.clipboard.writeText(jsonOutput.textContent).then(() => {
            btnCopy.textContent = 'Copiado!';
            setTimeout(() => btnCopy.textContent = 'Copiar JSON', 2000);
        });
    });
});
