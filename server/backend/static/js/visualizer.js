

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

    document.addEventListener('DOMContentLoaded', function () {

        document.querySelectorAll('.tabs .tab').forEach((btn, i) => {
            btn.addEventListener('click', () => window.showTab(i));
        });
        document.querySelectorAll('.progress-step').forEach((btn, i) => {
            btn.addEventListener('click', () => window.showStep(i));
        });

        if (!document.querySelector('.tabs .tab.active')) window.showTab(0);
        if (!document.querySelector('.progress-step.active')) window.showStep(0);
    });
})();