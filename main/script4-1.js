const SCALE = 400;
        const FOG_DENSITY = 0.00035;
        const CAM_MAX_DIST = 12000;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor(0x000005, 1);
        document.body.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x000005, FOG_DENSITY);

        const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 1, CAM_MAX_DIST);
        scene.add(camera);

        function makeStarLayer(count, size, opacity) {
            const verts = [];
            for (let i = 0; i < count; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 700 + Math.random() * 250;
                verts.push(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
            }
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
            const pts = new THREE.Points(geo, new THREE.PointsMaterial({
                color: 0xffffff, size, sizeAttenuation: false,
                depthTest: false, depthWrite: false, transparent: true, opacity
            }));
            pts.renderOrder = -999;
            camera.add(pts);
        }
        makeStarLayer(18000, 0.8, 0.18);
        makeStarLayer(2500, 1.5, 0.30);

        const nebulaColors = [
            { color: new THREE.Color(0.08, 0.04, 0.18), count: 3000, radius: 800, spread: 600 },
            { color: new THREE.Color(0.02, 0.06, 0.16), count: 2500, radius: -400, spread: 500 },
            { color: new THREE.Color(0.14, 0.04, 0.08), count: 2000, radius: 200, spread: 700 },
            { color: new THREE.Color(0.04, 0.10, 0.10), count: 1500, radius: 600, spread: 400 },
        ];

        nebulaColors.forEach(({ color, count, radius, spread }) => {
            const verts = [], cols = [];
            for (let i = 0; i < count; i++) {
                const u = (Math.random() - 0.5) * spread;
                const v = (Math.random() - 0.5) * spread * 0.4;
                const w = (Math.random() - 0.5) * spread;
                const dist = Math.sqrt(u * u + v * v + w * w) / spread;
                if (Math.random() > Math.exp(-dist * 2)) return;
                verts.push(u + radius, v, w);
                const jitter = 0.015;
                cols.push(
                    color.r + (Math.random() - 0.5) * jitter,
                    color.g + (Math.random() - 0.5) * jitter,
                    color.b + (Math.random() - 0.5) * jitter
                );
            }
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
            geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
            scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
                size: 3.5, sizeAttenuation: true,
                vertexColors: true, transparent: true, opacity: 0.55,
                depthWrite: false, blending: THREE.AdditiveBlending
            })));
        });

        const texCache = {};
        function glowTex(hex) {
            if (texCache[hex]) return texCache[hex];
            const size = 256, half = 128;
            const c = document.createElement('canvas');
            c.width = c.height = size;
            const ctx = c.getContext('2d');
            const n = parseInt((hex || 'fdfbf7').replace('#', ''), 16);
            const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
            const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
            grad.addColorStop(0, `rgba(${r},${g},${b},1)`);
            grad.addColorStop(0.15, `rgba(${r},${g},${b},0.8)`);
            grad.addColorStop(0.4, `rgba(${r},${g},${b},0.3)`);
            grad.addColorStop(0.7, `rgba(${r},${g},${b},0.08)`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, size, size);
            texCache[hex] = new THREE.CanvasTexture(c);
            return texCache[hex];
        }

        let nodeData = [], nodePoints = null, glowSprite = null;

        function buildNodes(data) {
            nodeData = data;
            const positions = new Float32Array(data.length * 3);
            const colors = new Float32Array(data.length * 3);
            const centroid = new THREE.Vector3();

            data.forEach((node, i) => {
                positions[i * 3] = node.x * SCALE;
                positions[i * 3 + 1] = node.y * SCALE;
                positions[i * 3 + 2] = node.z * SCALE;
                centroid.x += node.x * SCALE;
                centroid.y += node.y * SCALE;
                centroid.z += node.z * SCALE;
                const c = new THREE.Color(node.star_color || '#fdfbf7');
                colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
            });

            centroid.divideScalar(data.length);
            camTarget.copy(centroid);
            applyOrbit();

            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            nodePoints = new THREE.Points(geo, new THREE.PointsMaterial({
                size: 2.5, sizeAttenuation: false,
                vertexColors: true, depthWrite: false,
                transparent: true, opacity: 1.0
            }));
            scene.add(nodePoints);

            glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
                transparent: true, opacity: 0,
                depthWrite: false, blending: THREE.AdditiveBlending
            }));
            glowSprite.scale.set(80, 80, 1);
            scene.add(glowSprite);
        }

        let camTarget = new THREE.Vector3(0, 0, 0);
        let camDist = 4500;
        const camQuat = new THREE.Quaternion();
        camQuat.setFromEuler(new THREE.Euler(0.3, 0.4, 0));

        function applyOrbit() {
            const arm = new THREE.Vector3(0, 0, camDist).applyQuaternion(camQuat);
            camera.position.copy(camTarget).add(arm);
            camera.lookAt(camTarget);
        }
        applyOrbit();

        const velocity = { dx: 0, dy: 0 };
        const DAMPING = 0.88;
        let isDragging = false, isPan = false;
        let lastX = 0, lastY = 0, dragDist = 0;
        let animating = false;

        renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

        renderer.domElement.addEventListener('mousedown', e => {
            isDragging = true;
            isPan = e.button === 2 || e.shiftKey;
            lastX = e.clientX; lastY = e.clientY;
            dragDist = 0;
            velocity.dx = 0; velocity.dy = 0;
        });

        window.addEventListener('mousemove', e => {
            if (!isDragging) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            dragDist += Math.abs(dx) + Math.abs(dy);
            velocity.dx = dx; velocity.dy = dy;
            if (!animating) {
                if (isPan) applyPan(dx, dy);
                else applyRotate(dx, dy);
            }
        });

        window.addEventListener('mouseup', e => {
            const wasDrag = dragDist > 4;
            isDragging = false;
            if (wasDrag) return;
            if (!nodePoints) return;
            const mouse = new THREE.Vector2(
                (e.clientX / window.innerWidth) * 2 - 1,
                -(e.clientY / window.innerHeight) * 2 + 1
            );
            const ray = new THREE.Raycaster();
            ray.params.Points = { threshold: 8 };
            ray.setFromCamera(mouse, camera);
            const hits = ray.intersectObject(nodePoints);
            if (hits.length) selectNode(hits[0].index);
        });

        renderer.domElement.addEventListener('wheel', e => {
            camDist *= 1 + e.deltaY * 0.001;
            camDist = Math.max(50, Math.min(12000, camDist));
            if (!animating) applyOrbit();
        }, { passive: true });

        function applyRotate(dx, dy) {
            const speed = 0.004;
            const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -dx * speed);
            const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camQuat);
            const pitch = new THREE.Quaternion().setFromAxisAngle(right, -dy * speed);
            camQuat.premultiply(yaw).premultiply(pitch);
            applyOrbit();
        }

        function applyPan(dx, dy) {
            const speed = camDist * 0.0008;
            const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camQuat).multiplyScalar(-dx * speed);
            const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camQuat).multiplyScalar(dy * speed);
            camTarget.add(right).add(up);
            applyOrbit();
        }

        function tickInertia() {
            if (!isDragging && !animating && (Math.abs(velocity.dx) > 0.01 || Math.abs(velocity.dy) > 0.01)) {
                applyRotate(velocity.dx, velocity.dy);
                velocity.dx *= DAMPING;
                velocity.dy *= DAMPING;
            }
        }

        function selectNode(index) {
            const node = nodeData[index];
            const color = node.star_color || '#fdfbf7';
            const pos = new THREE.Vector3(node.x * SCALE, node.y * SCALE, node.z * SCALE);

            glowSprite.material.map = glowTex(color);
            glowSprite.material.opacity = 0.9;
            glowSprite.material.needsUpdate = true;
            glowSprite.position.copy(pos);

            const startTarget = camTarget.clone();
            const startDist = camDist;
            const endDist = 120;
            const t0 = performance.now();
            animating = true;
            velocity.dx = 0; velocity.dy = 0;

            (function fly(now) {
                const t = Math.min((now - t0) / 2200, 1);
                const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                camTarget.lerpVectors(startTarget, pos, ease);
                camDist = startDist + (endDist - startDist) * ease;
                applyOrbit();
                if (t < 1) requestAnimationFrame(fly);
                else animating = false;
            })(performance.now());

            showRecipeDetails(node);
        }

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        (function animate() {
            requestAnimationFrame(animate);
            tickInertia();
            renderer.render(scene, camera);
        })();

        fetch('./galaxy_data.json?v=' + Date.now())
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                buildNodes(data);
            })
            .catch(() => {
                document.getElementById('loading').innerText = 'ERROR LOADING DATA';
            });

        function safeParse(d) {
            if (!d) return ['Data unavailable'];
            if (Array.isArray(d)) return d;
            try { return JSON.parse(d.replace(/'/g, '"')); }
            catch (e) {
                const m = d.match(/'([^']+)'|"([^"]+)"/g);
                return m ? m.map(s => s.replace(/^['"]|['"]$/g, '')) : ['Parsing error'];
            }
        }

        function showRecipeDetails(node) {
            document.getElementById('info-panel').style.display = 'block';
            document.getElementById('recipe-name').innerText = node.name;
            const badge = document.getElementById('badge');
            badge.innerText = node.galaxy_cluster || 'UNKNOWN';
            badge.style.backgroundColor = node.star_color || '#888';
            const ingList = document.getElementById('ingredients-list');
            ingList.innerHTML = '';
            safeParse(node.ingredients).forEach(ing => {
                const li = document.createElement('li');
                li.innerText = ing.charAt(0).toUpperCase() + ing.slice(1);
                ingList.appendChild(li);
            });
            const stepsList = document.getElementById('steps-list');
            stepsList.innerHTML = '';
            safeParse(node.steps).forEach(step => {
                const li = document.createElement('li');
                li.innerText = step.charAt(0).toUpperCase() + step.slice(1);
                stepsList.appendChild(li);
            });
        }

        function closePanel() {
            document.getElementById('info-panel').style.display = 'none';
            if (glowSprite) glowSprite.material.opacity = 0;
        }