import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js'

const scene    = new THREE.Scene()
scene.background = null

const container = document.getElementById('threejs-box')
const W = container.clientWidth  || 480
const H = container.clientHeight || 280

const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000)
camera.position.z = 9

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
renderer.setSize(W, H)
container.appendChild(renderer.domElement)

scene.add(new THREE.AmbientLight(0xffffff, 0.4))
const pLight = new THREE.PointLight(0x00ffcc, 6, 30)
pLight.position.set(0, 0, 8)
scene.add(pLight)

const allNeurons = []
const allConnections = []
const allParticles = []

function createNeuron(x, y, z, color) {
    const geo  = new THREE.SphereGeometry(0.28, 24, 24)
    const mat  = new THREE.MeshStandardMaterial({
        color, emissive: color, emissiveIntensity: 1, roughness: 0.25
    })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.position.set(x, y, z)
    mesh.userData = { targetIntensity: 1, currentIntensity: 1 }
    scene.add(mesh)
    allNeurons.push(mesh)
    return mesh
}

const inputYs   = [-4.5, -3, -1.5, 0, 1.5, 3, 4.5]
const inputLayer = inputYs.map(y => createNeuron(-4, y, 0, 0x00d4ff))

const hiddenYs   = [-5.25, -3.75, -2.25, -0.75, 0.75, 2.25, 3.75, 5.25]
const hiddenLayer = hiddenYs.map((y, i) =>
    createNeuron(0, y, i % 2 === 0 ? -1 : 1, 0xff44cc))

const outputNeuron = createNeuron(4, 0, 0, 0x44ff88)

function makeParticle(startPos, endPos, color) {
    const geo  = new THREE.SphereGeometry(0.06, 6, 6)
    const mat  = new THREE.MeshBasicMaterial({ color })
    const mesh = new THREE.Mesh(geo, mat)
    scene.add(mesh)
    allParticles.push({ mesh, start: startPos.clone(), end: endPos.clone(), progress: Math.random() })
}

function connectNeurons(a, b, weight = 1) {
    const pts  = [a.position, b.position]
    const geo  = new THREE.BufferGeometry().setFromPoints(pts)
    const col  = weight >= 0 ? 0x00ffaa : 0xff3366
    const mat  = new THREE.LineBasicMaterial({
        color: col, transparent: true,
        opacity: Math.max(0.08, Math.min(0.7, Math.abs(weight) * 0.5))
    })
    const line = new THREE.Line(geo, mat)
    scene.add(line)
    makeParticle(a.position, b.position, col)
    allConnections.push({ line, neuronA: a, neuronB: b })
}

inputLayer.forEach(inp =>
    hiddenLayer.forEach(h => connectNeurons(inp, h, (Math.random() - 0.5))))

hiddenLayer.forEach(h => connectNeurons(h, outputNeuron, (Math.random() - 0.5)))

async function updateWeights() {
    const data = await fetch('/state').then(r => r.json())

    const wIH = data.weights_ih       
    const wHO = data.weights_ho.flat() 

    let ci = 0
    for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 8; j++) {
            const w    = wIH[i][j]
            const conn = allConnections[ci++]
            conn.line.material.color.set(w >= 0 ? 0x00ffaa : 0xff3366)
            conn.line.material.opacity = Math.max(0.08, Math.min(0.8, Math.abs(w) * 0.55))
        }
    }
    wHO.forEach((w, j) => {
        const conn = allConnections[ci++]
        if (!conn) return
        conn.line.material.color.set(w >= 0 ? 0x00ffaa : 0xff3366)
        conn.line.material.opacity = Math.max(0.08, Math.min(0.9, Math.abs(w) * 0.7))
    })

    const ho = data.hidden_output[0] 
    hiddenLayer.forEach((n, i) => {
        n.userData.targetIntensity = 0.4 + (ho[i] || 0) * 6
    })
}

const clock = new THREE.Clock()
let updateTimer = 0

function animate() {
    requestAnimationFrame(animate)
    const delta = clock.getDelta()
    const t     = clock.getElapsedTime()

    scene.rotation.y += 0.0015
    scene.rotation.x  = Math.sin(t * 0.4) * 0.08

    allNeurons.forEach(n => {
        const ud = n.userData
        ud.currentIntensity += (ud.targetIntensity - ud.currentIntensity) * 0.08
        n.material.emissiveIntensity = ud.currentIntensity +
            Math.sin(t * 2.5 + n.position.y) * 0.15
    })

    allParticles.forEach(p => {
        p.progress += delta * 0.35
        if (p.progress > 1) p.progress = 0
        p.mesh.position.lerpVectors(p.start, p.end, p.progress)
    })

    updateTimer += delta
    if (updateTimer > 0.5) {
        updateWeights()
        updateTimer = 0
    }

    renderer.render(scene, camera)
}

animate()

window.addEventListener('resize', () => {
    const w = container.clientWidth
    const h = container.clientHeight
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
})