class ModelComparisonTool {
    constructor() {
        this.selectedModels = [];
        this.maxSelections = 4;
        this.COLORS = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'];
    }

    toggleSelection(model) {
        const idx = this.selectedModels.findIndex(m => m.name === model.name);
        if (idx >= 0) {
            this.selectedModels.splice(idx, 1);
        } else {
            if (this.selectedModels.length >= this.maxSelections) {
                alert(`You can select max ${this.maxSelections} models`);
                return;
            }
            this.selectedModels.push(model);
        }
        this.renderComparison();
    }

    renderComparison() {
        // 1. Radar Chart - Overall capabilities
        this.renderRadarChart();

        // 4. Table - Detailed specs
        this.renderSpecsTable();
    }

    renderRadarChart() {
        if (this.selectedModels.length === 0) return;

        const categories = [
            'Reasoning', 'Coding', 'Math',
            'Language', 'Tool Use', 'Safety', 'Edge'
        ];

        // Ensure scores exist
        const datasets = this.selectedModels.map((model, idx) => ({
            label: model.name,
            data: categories.map(cat => {
                const key = cat.toLowerCase().replace(' ', '_');
                return (model.scores && model.scores[key]) || 0;
            }),
            borderColor: this.COLORS[idx],
            backgroundColor: this.COLORS[idx] + '33', // 20% opacity
            pointRadius: 4,
            borderWidth: 2
        }));

        const ctx = document.getElementById('radarChart');
        if (!ctx) return;

        if (window.radarChartInstance) {
            window.radarChartInstance.destroy();
        }

        window.radarChartInstance = new Chart(ctx.getContext('2d'), {
            type: 'radar',
            data: { labels: categories, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks: { stepSize: 20 },
                        pointLabels: {
                            font: { size: 12 }
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Capabilities Comparison'
                    },
                    legend: { position: 'top' }
                }
            }
        });
    }

    renderSpecsTable() {
        const table = document.getElementById('specsTable');
        if (!table) return;

        if (this.selectedModels.length === 0) {
            table.innerHTML = '<p class="text-center text-muted">Select models to compare specs</p>';
            return;
        }

        const specs = [
            { key: 'parameters', label: 'Parameters', format: (v) => v },
            { key: 'context_length', label: 'Context Window', format: (v) => v },
            { key: 'architecture', label: 'Architecture', format: (v) => v || '-' },
            { key: 'latency_ms', label: 'Latency', format: (v) => v ? v.toFixed(2) + ' ms' : '-' },
            { key: 'throughput_tps', label: 'Throughput', format: (v) => v ? v.toFixed(2) + ' t/s' : '-' },
            { key: 'memory_gb', label: 'Memory', format: (v) => v ? v.toFixed(2) + ' GB' : '-' },
            { key: 'energy_kwh', label: 'Energy/1k', format: (v) => v ? v.toFixed(4) + ' kWh' : '-' },
            { key: 'co2_kg', label: 'CO₂/1k', format: (v) => v ? v.toFixed(4) + ' kg' : '-' },
            { key: 'efficiency_score', label: 'Efficiency', format: (v) => v ? v.toFixed(1) : '-' }
        ];

        let html = '<table class="min-w-full divide-y divide-gray-200 shadow-sm rounded-lg overflow-hidden">';
        html += '<thead class="bg-gray-50"><tr><th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>';
        this.selectedModels.forEach(m => {
            html += `<th class="px-6 py-3 text-left text-xs font-medium text-gray-900 uppercase tracking-wider">${m.name}</th>`;
        });
        html += '</tr></thead><tbody class="bg-white divide-y divide-gray-200">';

        specs.forEach(spec => {
            html += `<tr><td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${spec.label}</td>`;
            this.selectedModels.forEach(m => {
                const value = m[spec.key];
                const formatted = spec.format(value);

                // Simple highlight logic
                const highlight = this.isBestValue(spec.key, value, this.selectedModels);
                const className = highlight ? 'bg-green-50 font-bold text-green-700' : 'text-gray-500';

                html += `<td class="px-6 py-4 whitespace-nowrap text-sm ${className}">${formatted}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        table.innerHTML = html;
    }

    isBestValue(key, value, models) {
        if (value === undefined || value === null || value === '-') return false;

        // Metrics where lower is better
        const lowerIsBetter = ['latency_ms', 'memory_gb', 'energy_kwh', 'co2_kg'];
        const values = models.map(m => m[key]).filter(v => v !== undefined && v !== null);

        if (values.length === 0) return false;

        const best = lowerIsBetter.includes(key)
            ? Math.min(...values)
            : Math.max(...values);

        return value === best;
    }
}

window.ModelComparisonTool = ModelComparisonTool;
