let autoTraining     = false
let trainingInterval = null
let stableEpochs     = 0
const STABILITY_REQUIRED = 30

const ctx   = document.getElementById('errorChart')
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Erro MSE (espaço normalizado)',
            data: [],
            borderColor: '#32a041',
            backgroundColor: 'rgba(50,160,65,0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true
        }]
    },
    options: {
        animation: false,
        plugins: { legend: { display: true } },
        scales: {
            x: { display: true, title: { display: true, text: 'Época' } },
            y: { display: true, title: { display: true, text: 'MSE' }, beginAtZero: true }
        }
    }
})

function updateInterface(data) {
    document.getElementById('epoch').innerText = data.epoch
    document.getElementById('error').innerText = data.error.toFixed(6)

    
    if (data.bias_h) {
        const bh = data.bias_h[0].map(v => v.toFixed(3)).join('  ')
        const bo = data.bias_o[0][0].toFixed(3)
        const el = document.getElementById('biasDisplay')
        if (el) el.innerText = `b¹ [ ${bh} ]   b² [ ${bo} ]`
    }

    const table = document.getElementById('tableBody')
    table.innerHTML = ''

    for (let i = 0; i < data.inputs.length; i++) {
        const entrada  = data.inputs[i].join(' | ')
        const esperado = data.expected[i].toFixed(0)
        const previsto = data.predictions[i].toFixed(1)
        const diff     = Math.abs(data.expected[i] - data.predictions[i])
        const pct      = (diff / data.expected[i]) * 100
        const emoji    = pct < 15 ? '✅' : pct < 30 ? '' : '❌'

        table.innerHTML += `
        <tr>
            <td class="mono">${entrada}</td>
            <td style="color:#32a041;">${esperado} kWh</td>
            <td>${previsto} kWh ${emoji}<br><small>${pct.toFixed(1)}% erro</small></td>
        </tr>`
    }

    chart.data.labels = data.history.map((_, i) => i + 1)
    chart.data.datasets[0].data = data.history
    chart.update('none')
}


const ALL_BTNS = () => document.querySelectorAll('button')

function setButtonsDisabled(disabled) {
    ALL_BTNS().forEach(b => { b.disabled = disabled })
}

function toggleAutoTrain() {
    autoTraining = !autoTraining
    const btn = document.getElementById('autoTrainBtn')

    if (autoTraining) {
        btn.innerText    = 'Parar Treinamento'
        stableEpochs     = 0
        trainingInterval = setInterval(nextEpoch, 30)
    } else {
        stopAutoTraining()
    }
}

function stopAutoTraining() {
    autoTraining = false
    clearInterval(trainingInterval)
    trainingInterval = null
    document.getElementById('autoTrainBtn').innerText = 'Treinamento Automático'
    stableEpochs = 0
}

async function nextEpoch() {
    const data = await fetch('/train', { method: 'POST' }).then(r => r.json())

    stableEpochs = data.error < 0.005 ? stableEpochs + 1 : 0
    updateInterface(data)

    if (autoTraining && stableEpochs >= STABILITY_REQUIRED) {
        stopAutoTraining()
        showToast('Rede convergiu! Treinamento encerrado.')
    }
}

async function turboTrain() {
    const EPOCHS = 5000
    const btn    = document.getElementById('turboBtn')

    if (autoTraining) stopAutoTraining()

    setButtonsDisabled(true)
    btn.innerText = ` Treinando ${EPOCHS.toLocaleString()} épocas…`

    const data = await fetch('/fast_train', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ epochs: EPOCHS })
    }).then(r => r.json())

    setButtonsDisabled(false)
    btn.innerText = ' Treino Turbo (5000 épocas)'

    updateInterface(data)
    showToast(`Turbo concluído! ${data.epoch} épocas — erro ${data.error.toFixed(6)}`)
}

async function resetNetwork() {
    const data = await fetch('/reset', { method: 'POST' }).then(r => r.json())
    updateInterface(data)
    showToast('Rede reiniciada com pesos aleatórios.')
}

async function corruptMemory() {
    const data = await fetch('/corrupt', { method: 'POST' }).then(r => r.json())
    updateInterface(data)
    showToast('Memória corrompida! Re-treine a rede.')
}

async function saveModel() {
    const result = await fetch('/save', { method: 'POST' }).then(r => r.json())
    if (result.ok) {
        showToast(`Modelo salvo! (época ${result.epoch})`)
    } else {
        showToast('Erro ao salvar.')
    }
}

async function loadModel() {
    const response = await fetch('/load', { method: 'POST' })
    if (!response.ok) {
        showToast('Nenhum modelo salvo encontrado.')
        return
    }
    const data = await response.json()
    updateInterface(data)
    showToast(`Modelo carregado! (época ${data.epoch})`)
}

async function estimarConsumo() {
    const body = {
        moradores:    parseFloat(document.getElementById('p_moradores').value)    || 0,
        tvs:          parseFloat(document.getElementById('p_tvs').value)          || 0,
        ventiladores: parseFloat(document.getElementById('p_ventiladores').value) || 0,
        geladeiras:   parseFloat(document.getElementById('p_geladeiras').value)   || 0,
        horas_ar:     parseFloat(document.getElementById('p_horas_ar').value)     || 0,
        lampadas:     parseFloat(document.getElementById('p_lampadas').value)     || 0,
        area:         parseFloat(document.getElementById('p_area').value)         || 0,
    }

    const result = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    }).then(r => r.json())

    document.getElementById('resultKwh').innerText   = result.kwh + ' kWh/mês'
    document.getElementById('resultReais').innerText = 'R$ ' + result.reais.toFixed(2)
    document.getElementById('resultNivel').innerText =
        result.kwh < 200 ? 'Baixo' : result.kwh < 450 ? 'Médio' : 'Alto'
}

function showToast(msg) {
    const t = document.createElement('div')
    t.className = 'toast'
    t.innerText = msg
    document.body.appendChild(t)
    setTimeout(() => t.classList.add('show'), 10)
    setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 400) }, 3000)
}

fetch('/state').then(r => r.json()).then(updateInterface)