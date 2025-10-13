

(function () {
    function activateList(list, idx, cls) {
        list.forEach((el, i) => el.classList.toggle(cls, i === idx));
    }

    window.showTab = function (index) {
        index = Number(index) || 0;
        const contents = Array.from(document.querySelectorAll('.tab-content'));
        const tabs = Array.from(document.querySelectorAll('.tab'));
        contents.forEach(c => c.classList.remove('active'));
        tabs.forEach(t => t.classList.remove('active'));
        const content = document.getElementById('tab' + index);
        if (content) content.classList.add('active');
        if (tabs[index]) tabs[index].classList.add('active');
    };

    window.showStep = function (index) {
        index = Number(index) || 0;
        const steps = Array.from(document.querySelectorAll('.step-container'));
        const progressSteps = Array.from(document.querySelectorAll('.progress-step'));
        steps.forEach(s => s.classList.remove('active'));
        progressSteps.forEach((s, i) => {
            s.classList.remove('active', 'completed');
            if (i < index) s.classList.add('completed');
            else if (i === index) s.classList.add('active');
        });
        const step = document.getElementById('step' + index);
        if (step) step.classList.add('active');
    };

    function bindVisualizerInteractions() {
        // 避免重复绑定：用 clone 替换节点
        document.querySelectorAll('.tabs .tab').forEach((btn) => {
            const clone = btn.cloneNode(true);
            btn.parentNode.replaceChild(clone, btn);
        });
        document.querySelectorAll('.progress-step').forEach((btn) => {
            const clone = btn.cloneNode(true);
            btn.parentNode.replaceChild(clone, btn);
        });

        document.querySelectorAll('.tabs .tab').forEach((btn, i) => {
            btn.addEventListener('click', () => window.showTab(i));
        });
        document.querySelectorAll('.progress-step').forEach((btn, i) => {
            btn.addEventListener('click', () => window.showStep(i));
        });

        if (!document.querySelector('.tabs .tab.active')) window.showTab(0);
        if (!document.querySelector('.progress-step.active')) window.showStep(0);
    }

    // 对外初始化接口，pageLoader 注入后调用
    window.initVisualizer = function (meta) {
        try {
            bindVisualizerInteractions();
            // 如需处理 meta，可在此读取 meta 并据此初始化
        } catch (e) {
            console.error('initVisualizer error', e);
        }
    };

    // 直接打开 HTML 时也初始化一次
    document.addEventListener('DOMContentLoaded', function () {
        try { window.initVisualizer && window.initVisualizer(); } catch (e) { /* ignore */ }
    });

})();