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
            console.log('Data loaded:', result);

            // Handle both array and object formats
            this.data = Array.isArray(result) ? result : (result.models || []);
            console.log('Parsed models:', this.data);

            // Update stats
            const countEl = document.getElementById('model-count');
            if (countEl) countEl.textContent = this.data.length;

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

        // Quantization filter
        document.getElementById('quant-filter').addEventListener('change', (e) => {
            this.filters.quantization = e.target.value;
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
        const categories = new Set();
        this.data.forEach(model => {
            if (model.categories) {
                model.categories.forEach(cat => categories.add(cat));
            }
        });

        const container = document.getElementById('category-tags');
        categories.forEach(category => {
            const tag = document.createElement('div');
            tag.className = 'tag';
            tag.textContent = category;
            tag.addEventListener('click', () => {
                tag.classList.toggle('active');
                if (this.filters.categories.has(category)) {
                    this.filters.categories.delete(category);
                } else {
                    this.filters.categories.add(category);
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
                    model.hf_repo.toLowerCase().includes(searchLower);

                if (!matchesSearch) return false;
            }

            // Size filter
            if (this.filters.size !== 'all') {
                const params = this.parseParameters(model.parameters);
                const sizeMatch = this.matchesSize(params, this.filters.size);
                if (!sizeMatch) return false;
            }

            // Quantization filter
            if (this.filters.quantization !== 'all') {
                const hasQuant = model.quantizations.some(q =>
                    q.name.toLowerCase().includes(this.filters.quantization)
                );
                if (!hasQuant) return false;
            }

            // Category filter
            if (this.filters.categories.size > 0) {
                const hasCategory = model.categories &&
                    model.categories.some(cat => this.filters.categories.has(cat));
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
                case 'reasoning':
                case 'coding':
                case 'math':
                case 'language':
                case 'edge':
                    aVal = a.scores?.[column] || 0;
                    bVal = b.scores?.[column] || 0;
                    break;
                case 'date':
                    aVal = new Date(a.date_added);
                    bVal = new Date(b.date_added);
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

    renderTable() {
        const tbody = document.getElementById('leaderboard-body');
        tbody.innerHTML = '';

        if (this.filteredData.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="11" style="text-align: center; padding: 2rem; color: var(--text-muted);">
                        No models match your filters. Try adjusting your search criteria.
                    </td>
                </tr>
            `;
            return;
        }

        this.filteredData.forEach((model, index) => {
            const row = document.createElement('tr');

            // Rank
            const rankClass = index < 3 ? `rank-${index + 1}` : '';
            row.innerHTML = `
                <td class="rank ${rankClass}">#${index + 1}</td>
                
                <td>
                    <a href="model.html?id=${model.id}" class="model-name-link">
                        <div class="model-name">${model.name}</div>
                    </a>
                    <div style="font-size: 0.875rem; color: var(--text-muted);">${model.family}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">
                        via <a href="https://huggingface.co/${model.hf_repo}" target="_blank" style="color: var(--speed-cyan);">HuggingFace</a>
                    </div>
                </td>
                
                <td>
                    <span class="badge badge-size">${model.parameters}</span>
                </td>
                
                <td>
                    <div class="score">${(model.aggregate_score || 0).toFixed(2)}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${model.aggregate_score || 0}%"></div>
                    </div>
                </td>
                
                <td>${this.formatScore(model.scores?.reasoning)}</td>
                <td>${this.formatScore(model.scores?.coding)}</td>
                <td>${this.formatScore(model.scores?.math)}</td>
                <td>${this.formatScore(model.scores?.language)}</td>
                <td>${this.formatScore(model.scores?.edge)}</td>
                
                <td>
                    ${(model.quantizations || []).map(q =>
                `<span class="badge badge-quant">${q.name}</span>`
            ).join(' ')}
                </td>
                
                <td>
                    <div style="display: flex; flex-direction: column; gap: 4px;">
                        <a href="model.html?id=${model.id}" class="btn-sm">📊 Details</a>
                        <a href="https://huggingface.co/${model.hf_repo}" target="_blank" class="btn-sm btn-outline">🤗 HF Repo</a>
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

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new LeaderboardManager();
});