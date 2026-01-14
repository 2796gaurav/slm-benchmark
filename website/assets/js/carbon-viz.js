class CarbonVisualization {
    renderEfficiencyScatterPlot(models) {
        const ctx = document.getElementById('efficiencyChart').getContext('2d');

        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Models',
                    data: models.map(m => {
                        const params = this.parseParameters(m.parameters);
                        return {
                            x: m.aggregate_score || 0,
                            y: m.energy_kwh || 0,
                            r: Math.max(5, Math.min(25, params / 100000000)),
                            model: m.name
                        };
                    }),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Accuracy Score' },
                        min: 0,
                        max: 100
                    },
                    y: {
                        title: { display: true, text: 'Energy (kWh/1k)' },
                        // reverse: true // Usually lower energy is better/higher up? Or standard plot?
                        // Let's keep standard and let bubble position speak.
                        min: 0
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Accuracy vs. Energy Efficiency'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const model = context.raw.model;
                                return [
                                    `Model: ${model}`,
                                    `Accuracy: ${context.raw.x.toFixed(1)}`,
                                    `Energy: ${context.raw.y?.toFixed(3) || 'N/A'} kWh`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }

    renderCarbonComparison(models) {
        // Bar chart showing CO2 emissions
        // Sort by CO2 asc (cleanest first)
        const sortedModels = [...models].sort((a, b) => (a.co2_kg || 0) - (b.co2_kg || 0)).slice(0, 10);

        const labels = sortedModels.map(m => m.name);
        const co2_data = sortedModels.map(m => m.co2_kg || 0);

        const ctx = document.getElementById('carbonChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'CO2 Emissions (kg/1k queries)',
                    data: co2_data,
                    backgroundColor: co2_data.map(val =>
                        val < 0.1 ? 'rgba(75, 192, 75, 0.7)' :
                            val < 0.3 ? 'rgba(255, 205, 86, 0.7)' : 'rgba(255, 99, 132, 0.7)'
                    )
                }]
            },
            options: {
                indexAxis: 'y',  // Horizontal bar
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Carbon Footprint (Top 10 Efficient)'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    parseParameters(paramStr) {
        if (!paramStr) return 0;
        if (typeof paramStr === 'number') return paramStr;
        const match = paramStr.toString().match(/([\d.]+)([KMB])/i);
        if (!match) return parseFloat(paramStr) || 0;

        const num = parseFloat(match[1]);
        const unit = match[2].toUpperCase();
        const multipliers = { K: 1e3, M: 1e6, B: 1e9 };
        return num * multipliers[unit];
    }
}
