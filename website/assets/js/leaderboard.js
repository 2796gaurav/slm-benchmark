// Leaderboard functionality for SLM Benchmark
// File: website/assets/js/leaderboard.js

class LeaderboardManager {
    constructor() {
        this.data = [];
        this.filteredData = [];
        this.currentSort = { column: 'rank', direction: 'asc' };
        this.filters = {
            search: '',
            size: 'all',
            quantization: 'all',
            categories: new Set()
        };

        this.init();
    }

    async init() {
        await this.loadData();
        this.setupEventListeners();
        this.renderCategoryTags();
        this.applyFilters();
    }

    async loadData() {
        try {
            console.log('Fetching leaderboard data...');
            const response = await fetch('./assets/data/leaderboard.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();

            // Handle both array and object formats
            this.data = Array.isArray(result) ? result : (result.models || []);

            // Update stats
            this.calculateStats();

            // Hide loading, show table
            const loadingEl = document.getElementById('loading');
            const tableEl = document.getElementById('leaderboard-table');
            if (loadingEl) loadingEl.style.display = 'none';
            if (tableEl) tableEl.style.display = 'table';

        } catch (error) {
            console.error('Failed to load leaderboard data:', error);
            const loadingEl = document.getElementById('loading');
            if (loadingEl) {
                loadingEl.innerHTML =
                    `<p style="color: var(--danger);">Failed to load data: ${error.message}. Please try again later.</p>`;
            }
        }
    }

    calculateStats() {
        if (!this.data || this.data.length === 0) return;

        // Top Model (hardware‑agnostic aggregate score)
        const topModel = [...this.data].sort((a, b) => (b.aggregate_score || 0) - (a.aggregate_score || 0))[0];
        document.getElementById('top-model-name').textContent = topModel.name;

        // Best Efficiency (purely informational, not used for ranking)
        const bestEff = [...this.data].sort((a, b) => (b.efficiency_score || 0) - (a.efficiency_score || 0))[0];
        document.getElementById('best-efficiency-model').textContent = bestEff.name;

        // Total Models
        document.getElementById('model-count-val').textContent = this.data.length;
    }

    setupEventListeners() {
        // Search
        document.getElementById('search').addEventListener('input', (e) => {
            this.filters.search = e.target.value.toLowerCase();
            this.applyFilters();
        });

        // Sort dropdown
        document.getElementById('sort').addEventListener('change', (e) => {
            this.sortData(e.target.value, 'desc');
        });

        // Size filter
        document.getElementById('size-filter').addEventListener('change', (e) => {
            this.filters.size = e.target.value;
            this.applyFilters();
        });

        // Table header sorting
        document.querySelectorAll('th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const column = th.dataset.sort;
                const direction = this.currentSort.column === column &&
                    this.currentSort.direction === 'asc' ? 'desc' : 'asc';
                this.sortData(column, direction);

                // Update UI
                document.querySelectorAll('th.sortable').forEach(h => {
                    h.classList.remove('sorted-asc', 'sorted-desc');
                });
                th.classList.add(direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
            });
        });
    }

    renderCategoryTags() {
        const baseCategories = {
            reasoning: { label: 'Reasoning', description: 'Logical and commonsense reasoning tasks (e.g. MMLU, ARC-Challenge, HellaSwag, TruthfulQA).' },
            coding: { label: 'Coding', description: 'Code generation and problem-solving tasks (e.g. HumanEval, MBPP).' },
            math: { label: 'Math', description: 'Multi-step quantitative reasoning (e.g. GSM8K, math QA).' },
            language: { label: 'Language', description: 'Language understanding and reading comprehension (e.g. BoolQ, PIQA, WinoGrande).' },
            safety: { label: 'Safety', description: 'Lightweight safety, bias, and truthfulness probes.' },
            edge: { label: 'Edge', description: 'Edge-centric metrics like latency and memory usage (informational only).' },
            efficiency: { label: 'Efficiency', description: 'Energy-efficiency and carbon metrics (informational only).' }
        };

        // Merge any custom categories from models to keep things future-proof
        this.data.forEach(model => {
            (model.categories || []).forEach(catRaw => {
                const key = catRaw.toLowerCase();
                if (!baseCategories[key]) {
                    baseCategories[key] = {
                        label: catRaw,
                        description: `${catRaw} category`
                    };
                }
            });
        });

        const container = document.getElementById('category-tags');
        if (!container) return;
        container.innerHTML = ''; // Clear old tags

        Object.entries(baseCategories).forEach(([key, meta]) => {
            const tag = document.createElement('button');
            tag.type = 'button';
            tag.className = 'tag';
            tag.title = `${meta.label}: ${meta.description}`;
            tag.setAttribute('aria-label', `${meta.label} category filter`);
            tag.textContent = meta.label;

            tag.addEventListener('click', () => {
                tag.classList.toggle('active');
                if (this.filters.categories.has(key)) {
                    this.filters.categories.delete(key);
                } else {
                    this.filters.categories.add(key);
                }
                this.applyFilters();
            });

            container.appendChild(tag);
        });
    }

    applyFilters() {
        this.filteredData = this.data.filter(model => {
            // Search filter
            if (this.filters.search) {
                const searchLower = this.filters.search;
                const matchesSearch =
                    model.name.toLowerCase().includes(searchLower) ||
                    model.family.toLowerCase().includes(searchLower) ||
                    (model.hf_repo && model.hf_repo.toLowerCase().includes(searchLower));

                if (!matchesSearch) return false;
            }

            // Size filter
            if (this.filters.size !== 'all') {
                const params = this.parseParameters(model.parameters);
                const sizeMatch = this.matchesSize(params, this.filters.size);
                if (!sizeMatch) return false;
            }

            // Quantization filter

            // Category filter
            if (this.filters.categories.size > 0) {
                const modelCategories = (model.categories || []).map(c => c.toLowerCase());
                const hasCategory = Array.from(this.filters.categories).some(cat => modelCategories.includes(cat));
                if (!hasCategory) return false;
            }

            return true;
        });

        this.renderTable();
    }

    sortData(column, direction) {
        this.currentSort = { column, direction };

        this.filteredData.sort((a, b) => {
            let aVal, bVal;

            switch (column) {
                case 'rank':
                    aVal = a.rank || 999;
                    bVal = b.rank || 999;
                    break;
                case 'model':
                    aVal = a.name.toLowerCase();
                    bVal = b.name.toLowerCase();
                    break;
                case 'parameters':
                    aVal = this.parseParameters(a.parameters);
                    bVal = this.parseParameters(b.parameters);
                    break;
                case 'aggregate':
                    aVal = a.aggregate_score || 0;
                    bVal = b.aggregate_score || 0;
                    break;
                case 'efficiency':
                    aVal = a.efficiency_score || 0;
                    bVal = b.efficiency_score || 0;
                    break;
                case 'co2':
                    aVal = a.co2_kg || 999;
                    bVal = b.co2_kg || 999;
                    break;
                case 'reasoning':
                case 'coding':
                case 'math':
                case 'language':
                case 'safety':
                case 'edge':
                    aVal = a.scores?.[column] || 0;
                    bVal = b.scores?.[column] || 0;
                    break;
                default:
                    aVal = a[column];
                    bVal = b[column];
            }

            if (typeof aVal === 'string') {
                return direction === 'asc' ?
                    aVal.localeCompare(bVal) :
                    bVal.localeCompare(aVal);
            }

            return direction === 'asc' ? aVal - bVal : bVal - aVal;
        });

        this.renderTable();
    }

    getBadgesForModel(model) {
        const emojis = [];
        const params = this.parseParameters(model.parameters);

        if (params > 0 && params <= 500e6) {
            emojis.push('🎯');
        }
        if (model.efficiency_score && model.efficiency_score >= 1000) {
            emojis.push('🌱');
        }
        if (model.scores && model.scores.safety && model.scores.safety >= 90) {
            emojis.push('');
        }
        return emojis.join('');
    }

    renderTable() {
        const tbody = document.getElementById('leaderboard-body');
        tbody.innerHTML = '';

        if (this.filteredData.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="13" style="text-align: center; padding: 3rem; color: var(--text-muted);">
                        No models match your filters. Try adjusting your search criteria.
                    </td>
                </tr>
            `;
            return;
        }

        this.filteredData.forEach((model, index) => {
            const row = document.createElement('tr');

            // Rank with badges
            const rankClass = index < 3 ? `rank-${index + 1}` : '';
            const badgeEmojis = this.getBadgesForModel(model);
            row.innerHTML = `
                <td class="rank ${rankClass}">
                    <div style="display: flex; align-items: center; gap: 0.4rem;">
                        <span>#${index + 1}</span>
                        <span style="font-size: 0.9rem; line-height: 1;">${badgeEmojis}</span>
                    </div>
                </td>
                
                <td>
                    <a href="model.html?id=${model.id}" class="model-name-link" style="text-decoration: none;">
                        <div class="model-name" style="font-size: 1.05rem;">${model.name}</div>
                    </a>
                    <div style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 2px;">
                        ${model.family} • <a href="https://huggingface.co/${model.hf_repo}" target="_blank" style="color: var(--speed-cyan); text-decoration: none;">HF Repository</a>
                    </div>
                </td>
                
                <td>
                    <span class="badge badge-size">${model.parameters}</span>
                </td>
                
                <td>
                    <div class="score" style="color: var(--speed-cyan); font-size: 1.1rem;">${(model.aggregate_score || 0).toFixed(2)}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${model.aggregate_score || 0}%"></div>
                    </div>
                </td>
                
                <td>${this.formatScore(model.scores?.reasoning)}</td>
                <td>${this.formatScore(model.scores?.coding)}</td>
                <td>${this.formatScore(model.scores?.math)}</td>
                <td>${this.formatScore(model.scores?.language)}</td>
                <td>${this.formatScore(model.scores?.safety)}</td>
                <td>${this.formatScore(model.scores?.edge)}</td>
                
                <td style="font-family: 'JetBrains Mono', monospace; color: var(--text-secondary);">
                    ${(model.co2_kg || 0).toFixed(4)}
                </td>

                <td>
                    <div style="display: flex; gap: 4px; flex-wrap: wrap;">
                        ${(model.quantizations || []).map(q =>
                `<span class="badge badge-quant">${q.name}</span>`
            ).join('')}
                    </div>
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    formatScore(score) {
        if (!score && score !== 0) return '-';
        return `<span style="font-weight: 600;">${score.toFixed(2)}</span>`;
    }

    parseParameters(paramStr) {
        if (!paramStr) return 0;
        const match = paramStr.toString().match(/([\d.]+)([KMB])/i);
        if (!match) return 0;

        const num = parseFloat(match[1]);
        const unit = match[2].toUpperCase();

        const multipliers = { K: 1e3, M: 1e6, B: 1e9 };
        return num * multipliers[unit];
    }

    matchesSize(params, size) {
        switch (size) {
            case 'tiny':
                return params < 100e6;
            case 'small':
                return params >= 100e6 && params < 500e6;
            case 'medium':
                return params >= 500e6 && params < 1e9;
            case 'large':
                return params >= 1e9 && params <= 3e9;
            default:
                return true;
        }
    }
}

// Ensure these are globally available even for strict CSP or module-like behavior
// Assuming CarbonVisualization and ModelComparisonTool are defined elsewhere or imported
window.CarbonVisualization = typeof CarbonVisualization !== 'undefined' ? CarbonVisualization : undefined;
window.ModelComparisonTool = typeof ModelComparisonTool !== 'undefined' ? ModelComparisonTool : undefined;

// Initialize on page load - v1.0.1 (forcing cache refresh)
document.addEventListener('DOMContentLoaded', () => {
    new LeaderboardManager();
});