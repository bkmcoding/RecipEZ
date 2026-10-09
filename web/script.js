const SCALE = 1000
const FOG_DENSITY = 0.00025
const CAM_MAX_DIST = 10000
const POINT_SIZE = 3

let currentFilter = 'ALL'
let isSearchActive = false
let activeLabelIndex = -1

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setClearColor(0x000005, 1)
document.body.appendChild(renderer.domElement)

const scene = new THREE.Scene()
scene.fog = new THREE.FogExp2(0x000005, FOG_DENSITY)
renderer.setClearColor(0x000005, 1)

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 1, CAM_MAX_DIST)
scene.add(camera)

function makeStarLayer(count, size, opacity) {
  const verts = []
  for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    const r = 700 + Math.random() * 250
    verts.push(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi))
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  const pts = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      color: 0xffffff,
      size,
      sizeAttenuation: false,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity,
    }),
  )
  pts.renderOrder = -999
  camera.add(pts)
}
makeStarLayer(18000, 0.8, 0.18)
makeStarLayer(2500, 1.5, 0.3)

const nebulaColors = [
  { color: new THREE.Color(0.08, 0.04, 0.18), count: 3000, radius: 800, spread: 600 },
  { color: new THREE.Color(0.02, 0.06, 0.16), count: 2500, radius: -400, spread: 500 },
  { color: new THREE.Color(0.14, 0.04, 0.08), count: 2000, radius: 200, spread: 700 },
  { color: new THREE.Color(0.04, 0.1, 0.1), count: 1500, radius: 600, spread: 400 },
]

nebulaColors.forEach(({ color, count, radius, spread }) => {
  const verts = [],
    cols = []
  for (let i = 0; i < count; i++) {
    const u = (Math.random() - 0.5) * spread
    const v = (Math.random() - 0.5) * spread * 0.4
    const w = (Math.random() - 0.5) * spread
    const dist = Math.sqrt(u * u + v * v + w * w) / spread
    if (Math.random() > Math.exp(-dist * 2)) return
    verts.push(u + radius, v, w)
    const jitter = 0.015
    cols.push(
      color.r + (Math.random() - 0.5) * jitter,
      color.g + (Math.random() - 0.5) * jitter,
      color.b + (Math.random() - 0.5) * jitter,
    )
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3))
  scene.add(
    new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 3.5,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    ),
  )
})

const texCache = {}
function glowTex(hex) {
  if (texCache[hex]) return texCache[hex]
  const size = 256,
    half = 128
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d')
  const n = parseInt((hex || 'fdfbf7').replace('#', ''), 16)
  const r = (n >> 16) & 255,
    g = (n >> 8) & 255,
    b = n & 255
  const grad = ctx.createRadialGradient(half, half, 0, half, half, half)
  grad.addColorStop(0, `rgba(${r},${g},${b},1)`)
  grad.addColorStop(0.15, `rgba(${r},${g},${b},0.8)`)
  grad.addColorStop(0.4, `rgba(${r},${g},${b},0.3)`)
  grad.addColorStop(0.7, `rgba(${r},${g},${b},0.08)`)
  grad.addColorStop(1, `rgba(${r},${g},${b},0)`)
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)
  texCache[hex] = new THREE.CanvasTexture(c)
  return texCache[hex]
}

let nodeData = [],
  nodePoints = null,
  glowSprite = null

function buildNodes(data) {
  nodeData = data
  const positions = new Float32Array(data.length * 3)
  const colors = new Float32Array(data.length * 3)
  const centroid = new THREE.Vector3()

  data.forEach((node, i) => {
    positions[i * 3] = node.x * SCALE
    positions[i * 3 + 1] = node.y * SCALE
    positions[i * 3 + 2] = node.z * SCALE
    centroid.x += node.x * SCALE
    centroid.y += node.y * SCALE
    centroid.z += node.z * SCALE
    const c = new THREE.Color(node.star_color || '#fdfbf7')
    colors[i * 3] = c.r
    colors[i * 3 + 1] = c.g
    colors[i * 3 + 2] = c.b
  })

  centroid.divideScalar(data.length)
  camTarget.copy(centroid)
  applyOrbit()

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const circleCanvas = document.createElement('canvas')
  circleCanvas.width = 32
  circleCanvas.height = 32
  const context = circleCanvas.getContext('2d')
  context.beginPath()
  context.arc(16, 16, 16, 0, 2 * Math.PI)
  context.fillStyle = 'white'
  context.fill()
  const circleTexture = new THREE.CanvasTexture(circleCanvas)

  nodePoints = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: POINT_SIZE,
      sizeAttenuation: true,
      vertexColors: true,
      depthWrite: false,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      map: circleTexture,
    }),
  )
  scene.add(nodePoints)

  glowSprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  )
  glowSprite.scale.set(80, 80, 1)
  scene.add(glowSprite)
  populateFilterDropdown()
}

let camTarget = new THREE.Vector3(0, 0, 0)
let camDist = 4500
const camQuat = new THREE.Quaternion()
camQuat.setFromEuler(new THREE.Euler(0.3, 0.4, 0))

function applyOrbit() {
  const arm = new THREE.Vector3(0, 0, camDist).applyQuaternion(camQuat)
  camera.position.copy(camTarget).add(arm)
  camera.lookAt(camTarget)
}
applyOrbit()

const velocity = { dx: 0, dy: 0 }
const DAMPING = 0.88
let isDragging = false,
  isPan = false
let lastX = 0,
  lastY = 0,
  dragDist = 0
let animating = false

renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault())

renderer.domElement.addEventListener('mousedown', (e) => {
  if (isSearchActive) return // <--- ADD THIS
  isDragging = true
  isPan = e.button === 2 || e.shiftKey
  lastX = e.clientX
  lastY = e.clientY
  dragDist = 0
  velocity.dx = 0
  velocity.dy = 0
})

window.addEventListener('mousemove', (e) => {
  if (!isDragging) return
  const dx = e.clientX - lastX
  const dy = e.clientY - lastY
  lastX = e.clientX
  lastY = e.clientY
  dragDist += Math.abs(dx) + Math.abs(dy)
  velocity.dx = dx
  velocity.dy = dy
  if (!animating) {
    if (isPan) applyPan(dx, dy)
    else applyRotate(dx, dy)
  }
})

window.addEventListener('mouseup', (e) => {
  if (e.target.tagName !== 'CANVAS') return
  const wasDrag = dragDist > 4
  isDragging = false
  if (wasDrag) return
  if (!nodePoints) return
  const mouse = new THREE.Vector2((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1)
  const ray = new THREE.Raycaster()
  ray.params.Points = { threshold: 8 }
  ray.setFromCamera(mouse, camera)

  const hits = ray.intersectObject(nodePoints)
  if (hits.length) {
    const validHit = hits.find((hit) => currentFilter === 'ALL' || nodeData[hit.index].galaxy_cluster === currentFilter)
    if (validHit) selectNode(validHit.index)
  }
})

renderer.domElement.addEventListener(
  'wheel',
  (e) => {
    camDist *= 1 + e.deltaY * 0.001
    camDist = Math.max(50, Math.min(12000, camDist))
    if (!animating) applyOrbit()
  },
  { passive: true },
)

function applyRotate(dx, dy) {
  const speed = 0.004
  const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -dx * speed)
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camQuat)
  const pitch = new THREE.Quaternion().setFromAxisAngle(right, -dy * speed)
  camQuat.premultiply(yaw).premultiply(pitch)
  applyOrbit()
}

function applyPan(dx, dy) {
  const speed = camDist * 0.0008
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camQuat).multiplyScalar(-dx * speed)
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camQuat).multiplyScalar(dy * speed)
  camTarget.add(right).add(up)
  applyOrbit()
}

function tickInertia() {
  if (!isDragging && !animating && (Math.abs(velocity.dx) > 0.01 || Math.abs(velocity.dy) > 0.01)) {
    applyRotate(velocity.dx, velocity.dy)
    velocity.dx *= DAMPING
    velocity.dy *= DAMPING
  }
}

function selectNode(index) {
  activeLabelIndex = index
  const node = nodeData[index]
  const color = node.star_color || '#fdfbf7'
  const pos = new THREE.Vector3(node.x * SCALE, node.y * SCALE, node.z * SCALE)

  glowSprite.material.map = glowTex(color)
  glowSprite.material.opacity = 0.9
  glowSprite.material.needsUpdate = true
  glowSprite.position.copy(pos)

  const startTarget = camTarget.clone()
  const startDist = camDist
  const endDist = 120
  const t0 = performance.now()
  animating = true
  velocity.dx = 0
  velocity.dy = 0
  ;(function fly(now) {
    const t = Math.min((now - t0) / 2200, 1)
    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
    camTarget.lerpVectors(startTarget, pos, ease)
    camDist = startDist + (endDist - startDist) * ease
    applyOrbit()
    if (t < 1) requestAnimationFrame(fly)
    else animating = false
  })(performance.now())

  showRecipeDetails(node)
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})
;(function animate() {
  requestAnimationFrame(animate)
  tickInertia()
  renderer.render(scene, camera)
})()

let activeEngine = 'tfidf'
let datasets = {
  sbert_all: null,
  sbert_ingredients: null,
  sbert_names: null,
  tfidf: null,
}
let isTransitioning = false

fetch('./model_data/tfidf_data.json?v=' + Date.now())
  .then((r) => r.json())
  .then((data) => {
    document.getElementById('loading').style.display = 'none'
    datasets.tfidf = data
    buildNodes(data)
  })
  .catch(() => {
    document.getElementById('loading').innerText = 'ERROR LOADING DATA'
  })

function switchEngine(targetEngine) {
  if (activeEngine === targetEngine || isTransitioning) return

  document.getElementById(`btn-${activeEngine}`).classList.remove('active')
  document.getElementById(`btn-${targetEngine}`).classList.add('active')
  activeEngine = targetEngine

  if (!datasets[targetEngine]) {
    document.getElementById('loading').innerText = 'LOADING DATA'
    document.getElementById('loading').style.display = 'block'

    const targetURL = './model_data/' + targetEngine + '_data.json?v=' + Date.now()

    fetch(targetURL)
      .then((r) => {
        if (!r.ok) throw new Error('HTTP ' + r.status)
        return r.json()
      })
      .then((data) => {
        document.getElementById('loading').style.display = 'none'
        datasets[targetEngine] = data
        transitionUniverse(datasets[targetEngine])
      })
      .catch((err) => {
        console.error('Fetch failed for URL:', targetURL, err)
        document.getElementById('loading').innerText = 'ERROR LOADING ' + targetEngine.toUpperCase()
      })
  } else {
    transitionUniverse(datasets[targetEngine])
  }
}

function transitionUniverse(targetData) {
  isTransitioning = true

  nodeData = targetData

  const positions = nodePoints.geometry.attributes.position.array

  const startCoords = new Float32Array(positions.length)
  const endCoords = new Float32Array(positions.length)

  for (let i = 0; i < targetData.length; i++) {
    startCoords[i * 3] = positions[i * 3]
    startCoords[i * 3 + 1] = positions[i * 3 + 1]
    startCoords[i * 3 + 2] = positions[i * 3 + 2]

    endCoords[i * 3] = targetData[i].x * SCALE
    endCoords[i * 3 + 1] = targetData[i].y * SCALE
    endCoords[i * 3 + 2] = targetData[i].z * SCALE
  }

  let camStartTarget = null
  let camEndTarget = null

  if (activeLabelIndex !== -1) {
    camStartTarget = camTarget.clone()
    camEndTarget = new THREE.Vector3(
      targetData[activeLabelIndex].x * SCALE,
      targetData[activeLabelIndex].y * SCALE,
      targetData[activeLabelIndex].z * SCALE,
    )
  }

  const duration = 2000
  const startTime = performance.now()

  function animateTransition(now) {
    let t = (now - startTime) / duration
    if (t >= 1) t = 1

    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t

    for (let i = 0; i < positions.length; i++) {
      positions[i] = startCoords[i] + (endCoords[i] - startCoords[i]) * ease
    }

    nodePoints.geometry.attributes.position.needsUpdate = true

    if (camStartTarget && camEndTarget) {
        camTarget.lerpVectors(camStartTarget, camEndTarget, ease);
        applyOrbit();

        if (glowSprite && glowSprite.material.opacity > 0) {
            glowSprite.position.copy(camTarget);
        }
    }

    if (t < 1) {
      requestAnimationFrame(animateTransition)
    } else {
      applyFilter(currentFilter)
      isTransitioning = false
    }
  }

  requestAnimationFrame(animateTransition)
}

function safeParse(d) {
  if (!d) return ['Data unavailable']
  if (Array.isArray(d)) return d
  try {
    return JSON.parse(d.replace(/'/g, '"'))
  } catch (e) {
    const m = d.match(/'([^']+)'|"([^"]+)"/g)
    return m ? m.map((s) => s.replace(/^['"]|['"]$/g, '')) : ['Parsing error']
  }
}

function showRecipeDetails(node) {
  document.getElementById('info-panel').style.display = 'block'
  document.getElementById('recipe-name').innerText = node.name

  const badge = document.getElementById('badge')
  badge.innerText = node.galaxy_cluster || 'UNKNOWN'

  const ingList = document.getElementById('ingredients-list')
  ingList.innerHTML = ''
  safeParse(node.ingredients).forEach((ing) => {
    const li = document.createElement('li')
    li.innerText = ing.charAt(0).toUpperCase() + ing.slice(1)
    ingList.appendChild(li)
  })

  const stepsList = document.getElementById('steps-list')
  stepsList.innerHTML = ''
  safeParse(node.steps).forEach((step) => {
    const li = document.createElement('li')
    li.innerText = step.charAt(0).toUpperCase() + step.slice(1)
    stepsList.appendChild(li)
  })

  const simList = document.getElementById('similar-list')
  simList.innerHTML = ''

  if (node.similar_recipes && node.similar_recipes.length > 0) {
    node.similar_recipes.forEach((simId) => {
      const simIndex = nodeData.findIndex((n) => n.id === simId)
      if (simIndex !== -1) {
        const simNode = nodeData[simIndex]
        const li = document.createElement('li')
        li.innerText = simNode.name

        li.style.cursor = 'pointer'
        li.style.textDecoration = 'underline'
        li.style.color = '#fff'
        li.style.marginBottom = '8px'

        li.onmouseover = () => (li.style.color = '#888')
        li.onmouseout = () => (li.style.color = '#fff')

        li.onclick = () => {
          selectNode(simIndex)
        }

        simList.appendChild(li)
      }
    })
  } else {
    simList.innerHTML = '<li style="color: #666; font-style: italic;">No semantic neighbors found.</li>'
  }
}

function closePanel() {
  document.getElementById('info-panel').style.display = 'none'
  if (glowSprite) glowSprite.material.opacity = 0
  activeLabelIndex = -1;
}

function openSearch() {
  isSearchActive = true
  document.getElementById('search-overlay').style.display = 'flex'
  document.getElementById('search-input').focus()
}

function closeSearch() {
  isSearchActive = false
  document.getElementById('search-overlay').style.display = 'none'
}

function closeSearchOnBg(e) {
  if (e.target.id === 'search-overlay') closeSearch()
}

function handleSearchKey(e) {
  if (e.key === 'Enter') executeSearch()
}

function handleSearchInput() {
  const query = document.getElementById('search-input').value.toLowerCase().trim()
  const resultsContainer = document.getElementById('search-results')
  resultsContainer.innerHTML = ''

  if (!query) return

  const searchTerms = query.split(/[\s,]+/).filter((term) => term.length > 2)
  if (searchTerms.length === 0) return

  let scoredNodes = nodeData.map((node, index) => {
    let score = 0
    const nodeName = (node.name || '').toLowerCase()
    const nodeIngredients = Array.isArray(node.ingredients)
      ? node.ingredients.join(' ').toLowerCase()
      : String(node.ingredients || '').toLowerCase()

    searchTerms.forEach((term) => {
      if (nodeIngredients.includes(term)) score += 1
      if (nodeName.includes(term)) score += 5
    })

    return { index, node, score }
  })

  let topMatches = scoredNodes
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10)

  topMatches.forEach((item) => {
    const li = document.createElement('li')
    li.style.padding = '12px'
    li.style.borderBottom = '1px solid #333'
    li.style.cursor = 'pointer'
    li.style.transition = 'background 0.2s'

    li.onmouseover = () => (li.style.background = 'rgba(255,255,255,0.1)')
    li.onmouseout = () => (li.style.background = 'transparent')

    li.innerHTML = `
                    <div style="color: #fff; font-weight: bold; font-size: 1.1rem; text-transform: capitalize;">${item.node.name}</div>
                    <div style="color: #888; font-size: 0.8rem; margin-top: 4px;">Sector: <span style="color: ${item.node.star_color}">${item.node.galaxy_cluster}</span></div>
                `

    li.onclick = () => {
      closeSearch()
      selectNode(item.index)
    }

    resultsContainer.appendChild(li)
  })
}

function toggleDropdown() {
  const list = document.getElementById('dropdown-list')
  list.style.display = list.style.display === 'flex' ? 'none' : 'flex'
}

document.addEventListener('click', (e) => {
  const dropdown = document.getElementById('custom-dropdown')
  const list = document.getElementById('dropdown-list')
  if (dropdown && !dropdown.contains(e.target)) {
    list.style.display = 'none'
  }
})

function populateFilterDropdown() {
  const uniqueClusters = [...new Set(nodeData.map((n) => n.galaxy_cluster))].sort()
  const list = document.getElementById('dropdown-list')
  list.innerHTML = ''

  const allOpt = document.createElement('div')
  allOpt.className = 'dropdown-item'
  allOpt.innerText = 'VIEW ALL RECIPES'
  allOpt.onclick = () => selectOption('ALL', 'VIEW ALL RECIPES')
  list.appendChild(allOpt)

  uniqueClusters.forEach((cluster) => {
    const opt = document.createElement('div')
    opt.className = 'dropdown-item'
    opt.innerText = cluster
    opt.onclick = () => selectOption(cluster, cluster)
    list.appendChild(opt)
  })
}

function selectOption(value, text) {
  document.getElementById('dropdown-header').innerText = text
  document.getElementById('dropdown-list').style.display = 'none'
  applyFilter(value)
}

function applyFilter(selectedCluster) {
  currentFilter = selectedCluster
  if (!nodePoints) return

  const colors = nodePoints.geometry.attributes.color.array

  nodeData.forEach((node, i) => {
    if (selectedCluster === 'ALL' || node.galaxy_cluster === selectedCluster) {
      const c = new THREE.Color(node.star_color || '#fdfbf7')
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
    } else {
      colors[i * 3] = 0
      colors[i * 3 + 1] = 0
      colors[i * 3 + 2] = 0
    }
  })

  nodePoints.geometry.attributes.color.needsUpdate = true
}

let currentSearchMode = 'recipe'

function setSearchMode(mode) {
  currentSearchMode = mode
  const input = document.getElementById('search-input')

  document.getElementById('toggle-recipe').classList.remove('active')
  document.getElementById('toggle-ingredient').classList.remove('active')
  document.getElementById(`toggle-${mode}`).classList.add('active')

  if (mode === 'recipe') {
    input.placeholder = 'Enter recipe name (e.g., Beef Wellington)...'
  } else {
    input.placeholder = 'Enter ingredients (e.g., chicken, garlic, ginger)...'
  }

  document.getElementById('search-results').innerHTML = ''
  input.value = ''
  input.focus()
}

function handleLiveSearch() {
  if (currentSearchMode !== 'recipe') return

  const query = document.getElementById('search-input').value.toLowerCase().trim()
  const resultsContainer = document.getElementById('search-results')
  resultsContainer.innerHTML = ''

  if (!query) return

  // Score and sort for live recipe names
  let scoredNodes = nodeData.map((node, index) => {
    let score = 0
    const nodeName = (node.name || '').toLowerCase()
    if (nodeName.includes(query)) score += 100 - nodeName.length
    return { index, node, score }
  })

  let topMatches = scoredNodes
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10)

  topMatches.forEach((item) => {
    resultsContainer.innerHTML += createResultListItem(item.node, item.index, false)
  })
}

function executeSearch() {
  const query = document.getElementById('search-input').value.toLowerCase().trim()
  if (!query) return

  if (currentSearchMode === 'recipe') {
    const firstResult = document.querySelector('#search-results li')
    if (firstResult) firstResult.click()
  } else {
    runKNNSearch(query)
  }
}

function runKNNSearch(query) {
  const searchTerms = query.toLowerCase().split(/[\s,]+/).filter(term => term.length > 2);
  const resultsContainer = document.getElementById('search-results');
  
  if (searchTerms.length === 0) return;

  let bestSeedIdx = -1;
  let highestScore = -Infinity;

  // 1. O(N) Weighted Lexical Scan
  nodeData.forEach((node, index) => {
      const ings = Array.isArray(node.ingredients) 
          ? node.ingredients.join(" ").toLowerCase() 
          : String(node.ingredients || "").toLowerCase();
      
      let matchCount = 0;
      
      searchTerms.forEach(term => {
          // Strict word boundary to prevent "ham" matching "graham"
          const regex = new RegExp(`\\b${term}\\b`, 'i');
          if (regex.test(ings)) {
              matchCount++;
          }
      });

      // 2. The Scoring Algorithm
      if (matchCount > 0) {
          // Base points: Huge reward for matching multiple requested ingredients
          let score = matchCount * 100; 
          
          // Penalty: Count roughly how many ingredients are in the recipe (via commas)
          const ingredientArrayLength = ings.split(',').length; 
          
          // Subtract points for bloat. A 5-ingredient recipe loses 5 points. A 25-ingredient recipe loses 25 points.
          score -= ingredientArrayLength; 

          if (score > highestScore) {
              highestScore = score;
              bestSeedIdx = index;
          }
      }
  });

  // 3. Render the Results
  if (bestSeedIdx !== -1) {
      const seedNode = nodeData[bestSeedIdx];
      let resultsHTML = `<li style="color: #666; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 4px;">Optimal Seed Match:</li>`;

      // Render Seed
      resultsHTML += createResultListItem(seedNode, bestSeedIdx, true);

      // Render pre-computed Semantic Neighbors
      if (seedNode.similar_recipes && seedNode.similar_recipes.length > 0) {
          seedNode.similar_recipes.forEach(simId => {
              const simIdx = nodeData.findIndex(n => n.id === simId);
              if (simIdx !== -1) {
                  resultsHTML += createResultListItem(nodeData[simIdx], simIdx, false);
              }
          });
      } else {
          resultsHTML += `<li style="color: #666; font-style: italic; padding: 10px;">No semantic neighbors computed for this seed.</li>`;
      }

      resultsContainer.innerHTML = resultsHTML;
  } else {
      resultsContainer.innerHTML = '<li style="color: #ff5555; padding: 16px;">Error: No recipes found containing those ingredients.</li>';
  }
}

function createResultListItem(node, index, isSeed) {
  const borderStyle = isSeed ? 'border-left: 3px solid #fff;' : 'border-left: 3px solid transparent;'
  const bgHover = isSeed ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.08)'

  return `
                <li style="padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); ${borderStyle} cursor: pointer; transition: all 0.2s;"
                    onmouseover="this.style.background='${bgHover}'" 
                    onmouseout="this.style.background='transparent'"
                    onclick="closeSearch(); selectNode(${index});">
                    <div style="color: #fff; font-weight: bold; font-size: 1.05rem; text-transform: capitalize;">
                        ${isSeed ? '⭐ ' : ''}${node.name}
                    </div>
                    <div style="color: #888; font-size: 0.8rem; margin-top: 4px;">
                        Sector: <span style="color: ${node.star_color}">${node.galaxy_cluster}</span>
                    </div>
                </li>
            `
}
