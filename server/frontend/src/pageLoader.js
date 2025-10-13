// ...existing code...
(function () {
    async function loadPage(cluster = 'default', name = 'index') {
        try {
            const metaRes = await fetch(`/api/pages/${encodeURIComponent(cluster)}/${encodeURIComponent(name)}`);
            if (!metaRes.ok) throw new Error('meta not found');
            const meta = await metaRes.json();
            const htmlRes = await fetch(`/src/html/${meta.file}`);
            if (!htmlRes.ok) throw new Error('template not found');
            const html = await htmlRes.text();

            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');


            const headElems = Array.from(doc.head.querySelectorAll('style, link[rel="stylesheet"]'));
            headElems.forEach(el => {
                if (el.tagName.toLowerCase() === 'link') {
                    const href = el.getAttribute('href');
                    if (href && !document.querySelector(`link[rel="stylesheet"][href="${href}"]`)) {
                        const link = document.createElement('link');
                        link.rel = 'stylesheet';
                        link.href = href;
                        document.head.appendChild(link);
                    }
                } else {
                    const style = document.createElement('style');
                    style.textContent = el.textContent;
                    document.head.appendChild(style);
                }
            });

            const scripts = Array.from(doc.querySelectorAll('script'));
            const container = document.getElementById('app');
            container.innerHTML = '';
            Array.from(doc.body.childNodes).forEach(node => {
                if (node.tagName && node.tagName.toLowerCase() === 'nav') return;
                container.appendChild(document.importNode(node, true));
            });

            const addScript = (src, type, isDefer) => new Promise((resolve, reject) => {
                if (src && document.querySelector(`script[src="${src}"]`)) return setTimeout(resolve, 0);
                const s = document.createElement('script');
                if (type) s.type = type;
                if (src) s.src = src;
                if (isDefer) s.defer = true;
                if (!type || type === 'text/javascript') s.async = false;
                s.onload = () => resolve();
                s.onerror = () => reject(new Error('Failed to load script ' + src));
                (isDefer || type === 'module' ? document.head : document.body).appendChild(s);
            });

            const loadExternalScript = (src, type, attrs = {}) => new Promise((resolve, reject) => {
                if (document.querySelector(`script[src="${src}"]`)) return setTimeout(resolve, 0);
                const s = document.createElement('script');
                if (type) s.type = type;
                s.src = src;
                // keep order for classic scripts
                if (!type || type === 'text/javascript') s.async = false;
                Object.keys(attrs).forEach(k => s.setAttribute(k, attrs[k]));
                s.onload = () => resolve();
                s.onerror = () => reject(new Error('Failed to load script ' + src));
                document.body.appendChild(s);
            });

            for (const sEl of scripts) {
                const src = sEl.getAttribute('src');
                const type = sEl.getAttribute('type');
                const attrs = {};
                if (sEl.hasAttribute('defer')) attrs.defer = '';
                if (src) {
                    await addScript(src, type, attrs);
                } else {
                    const inline = document.createElement('script');
                    if (type) inline.type = type;
                    inline.text = sEl.textContent;
                    document.body.appendChild(inline);
                }
            }

            if (typeof window.initPage === 'function') {
                try { window.initPage(meta); } catch (e) { console.error('initPage error', e); }
            }
            if (typeof window.initVisualizer === 'function') {
                try { window.initVisualizer(meta); } catch (e) { console.error('initVisualizer error', e); }
            }

            history.pushState({ cluster, name }, '', `/page/${cluster}/${name}`);
        } catch (e) {
            console.error(e);
            const el = document.getElementById('app');
            if (el) el.innerHTML = `<p>Error: ${e.message}</p>`;
        }
    }

    window.linkToPage = function (el) {
        const cluster = el.dataset.cluster;
        const name = el.dataset.name;
        loadPage(cluster, name);
    };

    window.addEventListener('popstate', function (e) {
        if (e.state) loadPage(e.state.cluster, e.state.name);
    });

    document.addEventListener('DOMContentLoaded', function () {
        const parts = location.pathname.split('/').filter(Boolean);
        if (parts[0] === 'page' && parts[1] && parts[2]) {
            loadPage(parts[1], parts[2]);
        }
        else {
            loadPage('default', 'index');
        }
    });
})();