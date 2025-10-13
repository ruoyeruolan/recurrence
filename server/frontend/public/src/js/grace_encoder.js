// 生成高斯分布的概率密度函数
(function () {
    function gaussianPDF(x, mu, sigma) {
        return (1 / (sigma * Math.sqrt(2 * Math.PI))) *
            Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
    }

    // 从高斯分布采样
    function sampleGaussian(mu, sigma) {
        // Box-Muller 变换
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return mu + sigma * z;
    }

    // 初始化图表
    let chart;
    let samples = [];

    function initChart() {
        const ctx = document.getElementById('distributionChart').getContext('2d');

        const xValues = [];
        for (let x = -10; x <= 10; x += 0.1) {
            xValues.push(x);
        }

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: xValues,
                datasets: [{
                    label: '概率密度函数 N(z|μ,σ²)',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3
                }, {
                    label: '采样点',
                    data: [],
                    type: 'scatter',
                    backgroundColor: '#ff6b6b',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '概率密度',
                            font: { size: 14, weight: 'bold' }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'z 值',
                            font: { size: 14, weight: 'bold' }
                        },
                        ticks: {
                            callback: function (value, index, ticks) {
                                // value 是索引，需要转换为实际的 x 值
                                const actualValue = this.getLabelForValue(value);
                                return parseFloat(actualValue).toFixed(2);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { font: { size: 12 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(4);
                            }
                        }
                    }
                }
            }
        });
    }

    function updateChart() {
        const mu = parseFloat(document.getElementById('muSlider').value);
        const logvar = parseFloat(document.getElementById('logvarSlider').value);
        const sigma = Math.exp(0.5 * logvar);

        // 更新显示值
        document.getElementById('muValue').textContent = mu.toFixed(2);
        document.getElementById('logvarValue').textContent = logvar.toFixed(2);
        document.getElementById('sigmaValue').textContent = sigma.toFixed(2);

        // 计算PDF
        const xValues = chart.data.labels;
        const pdfValues = xValues.map(x => gaussianPDF(x, mu, sigma));

        chart.data.datasets[0].data = pdfValues;

        // 更新采样点
        const samplePoints = samples.map(s => ({ x: s, y: gaussianPDF(s, mu, sigma) }));
        chart.data.datasets[1].data = samplePoints;

        chart.update();
    }

    function sampleOnce() {
        const mu = parseFloat(document.getElementById('muSlider').value);
        const logvar = parseFloat(document.getElementById('logvarSlider').value);
        const sigma = Math.exp(0.5 * logvar);

        const sample = sampleGaussian(mu, sigma);
        samples.push(sample);

        updateSamplesDisplay();
        updateChart();
    }

    function sampleMultiple() {
        const mu = parseFloat(document.getElementById('muSlider').value);
        const logvar = parseFloat(document.getElementById('logvarSlider').value);
        const sigma = Math.exp(0.5 * logvar);

        for (let i = 0; i < 100; i++) {
            const sample = sampleGaussian(mu, sigma);
            samples.push(sample);
        }

        updateSamplesDisplay();
        updateChart();
    }

    function clearSamples() {
        samples = [];
        updateSamplesDisplay();
        updateChart();
    }

    function updateSamplesDisplay() {
        const display = document.getElementById('samplesDisplay');

        if (samples.length === 0) {
            display.textContent = '采样结果将显示在这里...';
            return;
        }

        const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
        const variance = samples.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / samples.length;
        const std = Math.sqrt(variance);

        display.innerHTML = `
                <strong>采样统计 (共 ${samples.length} 个样本):</strong><br>
                样本均值: ${mean.toFixed(4)}<br>
                样本标准差: ${std.toFixed(4)}<br>
                最近5个样本: ${samples.slice(-5).map(s => s.toFixed(3)).join(', ')}
            `;
    }

    // 初始化
    // window.onload = function() {
    //     initChart();
    //     updateChart();

    //     document.getElementById('muSlider').addEventListener('input', updateChart);
    //     document.getElementById('logvarSlider').addEventListener('input', updateChart);
    // };

    function attachChartListeners() {
        const muSlider = document.getElementById('muSlider');
        const logvarSlider = document.getElementById('logvarSlider');
        if (muSlider) muSlider.addEventListener('input', updateChart);
        if (logvarSlider) logvarSlider.addEventListener('input', updateChart);
    }

    // 可由 pageLoader 调用的初始化接口（动态注入时使用）
    window.initVisualizer = function (meta) {
        try {
            // 如果 initChart/updateChart 在外部脚本中，确保外部脚本已加载后调用
            if (typeof initChart === 'function') initChart();
            if (typeof updateChart === 'function') updateChart();
            attachChartListeners();
        } catch (e) {
            console.error('initVisualizer error', e);
        }
    };

    // 仍兼容直接打开 HTML 的场景
    window.onload = function () {
        if (typeof window.initVisualizer === 'function') {
            window.initVisualizer();
        } else {
            // fallback，如果外部没有 initVisualizer
            try {
                if (typeof initChart === 'function') initChart();
                if (typeof updateChart === 'function') updateChart();
                attachChartListeners();
            } catch (e) {
                console.error(e);
            }
        }
    };
})();