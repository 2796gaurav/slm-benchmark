// SLM Marketplace - Discovery Engine
// File: website/assets/js/marketplace.js

class MarketplaceManager {
    constructor() {
        this.data = [];
        this.filteredData = [];
        this.selectedUseCase = 'all';
        this.filters = {
            search: '',
            ram: 'all',
            context: 'all',
            size: 'all'
        };
        this.sortBy = 'aggregate';

        this.init();
    }

    async init() {
        await this.loadData();
        this.setupEventListeners();
        this.applyFilters();
    }

    async loadData() {
        try {
            const response = await fetch('./assets/data/leaderboard.json');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const result = await response.json();
            this.data = Array.isArray(result) ? result : (result.models || []);
            this.updateStats();
            this.hideLoading();
        } catch (error) {
            console.error('Failed to load marketplace data:', error);
            document.getElementById('loading').innerHTML =
                `<p style="color: var(--danger);">Failed to load data: ${error.message}</p>`;
        }
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('model-grid').style.display = 'grid';
    }

    updateStats() {
        document.getElementById('model-count').textContent = this.data.length;

        // Top model by aggregate
        const topModel = [...this.data].sort((a, b) => (b.aggregate_score || 0) - (a.aggregate_score || 0))[0];
        if (topModel) {
            document.getElementById('top-model-name').textContent = topModel.name;
        }

        // Fastest model (best TPS in Q4)
        const fastestModel = [...this.data].sort((a, b) => {
            const aTps = a.performance?.quantizations?.q4?.tps_output || 0;
            const bTps = b.performance?.quantizations?.q4?.tps_output || 0;
            return bTps - aTps;
        })[0];
        if (fastestModel) {
            document.getElementById('fastest-model').textContent = fastestModel.name;
        }
    }

    setupEventListeners() {
        // Use case cards
        document.querySelectorAll('.use-case-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectedUseCase = card.dataset.usecase;
                document.querySelectorAll('.use-case-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                this.updateResultsTitle();
                this.applyFilters();
            });
        });

        // Filters
        document.getElementById('search').addEventListener('input', (e) => {
            this.filters.search = e.target.value.toLowerCase();
            this.applyFilters();
        });

        // Semantic search suggestion chips (optional helpers)
        document.querySelectorAll('.search-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query || chip.textContent;
                const searchInput = document.getElementById('search');
                if (searchInput) {
                    searchInput.value = query;
                }
                this.filters.search = query.toLowerCase();
                this.applyFilters();
                if (searchInput) {
                    searchInput.focus();
                }
            });
        });

        document.getElementById('ram-filter').addEventListener('change', (e) => {
            this.filters.ram = e.target.value;
            this.applyFilters();
        });

        document.getElementById('context-filter').addEventListener('change', (e) => {
            this.filters.context = e.target.value;
            this.applyFilters();
        });

        document.getElementById('size-filter').addEventListener('change', (e) => {
            this.filters.size = e.target.value;
            this.applyFilters();
        });

        // Sort
        document.getElementById('sort').addEventListener('change', (e) => {
            this.sortBy = e.target.value;
            this.applyFilters();
        });
    }

    updateResultsTitle() {
        const titles = {
            'all': 'All Models',
            'rag': 'Best for RAG & Q&A',
            'function_calling': 'Best for Function Calling',
            'coding': 'Best for Coding',
            'reasoning': 'Best for Reasoning',
            'guardrails': 'Best for Safety & Guardrails'
        };
        document.getElementById('results-title').textContent = titles[this.selectedUseCase] || 'Models';
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
            case 'tiny': return params < 500e6;
            case 'small': return params >= 500e6 && params < 1.5e9;
            case 'medium': return params >= 1.5e9 && params < 3e9;
            case 'large': return params >= 3e9 && params <= 5e9;
            default: return true;
        }
    }

    tokenizeQuery(query) {
        if (!query) return [];
        return query
            .toLowerCase()
            .split(/[^a-z0-9+]+/g)
            .map(t => t.trim())
            .filter(Boolean);
    }

    computeRelevance(model, query) {
        const tokens = this.tokenizeQuery(query);
        if (!tokens.length) return 0;

        let score = 0;

        const safe = (v) => (v ? v.toString().toLowerCase() : '');

        const fields = {
            name: 7,
            family: 6,
            hf_repo: 5,
            parameters: 2,
            architecture: 3,
            license: 1
        };

        const bestForText = (model.best_for || []).join(' ').toLowerCase();
        const tagsText = (model.tags || []).join(' ').toLowerCase();
        const categoriesText = (model.categories || []).join(' ').toLowerCase();

        const useCaseLabels = {
            rag: 'rag retrieval qa question answering documents context',
            function_calling: 'function calling tools apis agent agentic',
            coding: 'code coding programming dev developer',
            reasoning: 'reasoning logic mmlu commonsense',
            guardrails: 'guardrails safety moderation toxicity'
        };

        const q8 = model.performance?.quantizations?.q8 || {};
        const q4 = model.performance?.quantizations?.q4 || {};

        tokens.forEach(token => {
            // Direct field matches (name, family, etc.)
            Object.entries(fields).forEach(([key, weight]) => {
                const val = safe(model[key]);
                if (val && val.includes(token)) {
                    score += weight;
                }
            });

            // Best-for and tags text
            if (bestForText && bestForText.includes(token)) {
                score += 4;
            }
            if (tagsText && tagsText.includes(token)) {
                score += 3;
            }
            if (categoriesText && categoriesText.includes(token)) {
                score += 2;
            }

            // Use case intent → reward models that are strong in that use case
            Object.entries(useCaseLabels).forEach(([useCaseKey, labelText]) => {
                if (!labelText.includes(token)) return;
                const uc = model.use_cases?.[useCaseKey];
                if (uc) {
                    score += (uc.score || 0) / 15; // scale scores into relevance
                    if (uc.recommended) score += 5;
                }
            });

            // Performance-oriented intents
            const perfKeywords = ['fast', 'speed', 'latency', 'throughput', 'tps'];
            if (perfKeywords.includes(token)) {
                const tps = q4.tps_output || q8.tps_output || 0;
                score += tps / 10;
            }

            const lowResourceKeywords = ['edge', 'mobile', 'raspberry', 'pi', 'low', 'lightweight', 'small', 'ram', 'memory'];
            if (lowResourceKeywords.includes(token)) {
                const ram = q8.ram_gb || q4.ram_gb || 0;
                if (ram > 0) {
                    score += (16 - Math.min(ram, 16)); // favor lower RAM usage
                }
                const params = this.parseParameters(model.parameters);
                if (params > 0) {
                    score += (3e9 - Math.min(params, 3e9)) / 5e8; // favor smaller models
                }
            }
        });

        return score;
    }

    applyFilters() {
        this.filteredData = this.data.filter(model => {
            // Semantic search filter
            if (this.filters.search) {
                const relevance = this.computeRelevance(model, this.filters.search);
                model._relevance = relevance;
                if (relevance <= 0) return false;
            } else {
                model._relevance = 0;
            }

            // RAM filter (based on Q8 quantization)
            if (this.filters.ram !== 'all') {
                const maxRam = parseFloat(this.filters.ram);
                const modelRam = model.performance?.quantizations?.q8?.ram_gb ||
                    model.performance?.quantizations?.q4?.ram_gb || 999;
                if (modelRam > maxRam) return false;
            }

            // Context filter
            if (this.filters.context !== 'all') {
                const minContext = parseInt(this.filters.context);
                const modelContext = model.context_window || 0;
                if (modelContext < minContext) return false;
            }

            // Size filter
            if (this.filters.size !== 'all') {
                const params = this.parseParameters(model.parameters);
                if (!this.matchesSize(params, this.filters.size)) return false;
            }

            // Use case filter (only show models recommended for use case OR all models if score > 50)
            if (this.selectedUseCase !== 'all') {
                const useCaseData = model.use_cases?.[this.selectedUseCase];
                if (!useCaseData || useCaseData.score < 50) return false;
            }

            return true;
        });

        // Sort
        this.sortData();
        this.renderCards();
    }

    sortData() {
        this.filteredData.sort((a, b) => {
            // When a search query is active, prioritize semantic relevance
            if (this.filters.search) {
                const aRel = a._relevance || 0;
                const bRel = b._relevance || 0;
                if (bRel !== aRel) {
                    return bRel - aRel;
                }
            }

            switch (this.sortBy) {
                case 'aggregate':
                    return (b.aggregate_score || 0) - (a.aggregate_score || 0);
                case 'rag':
                    return (b.use_cases?.rag?.score || 0) - (a.use_cases?.rag?.score || 0);
                case 'function_calling':
                    return (b.use_cases?.function_calling?.score || 0) - (a.use_cases?.function_calling?.score || 0);
                case 'coding':
                    return (b.use_cases?.coding?.score || 0) - (a.use_cases?.coding?.score || 0);
                case 'reasoning':
                    return (b.use_cases?.reasoning?.score || 0) - (a.use_cases?.reasoning?.score || 0);
                case 'tps':
                    return (b.performance?.quantizations?.q4?.tps_output || 0) -
                        (a.performance?.quantizations?.q4?.tps_output || 0);
                case 'ram':
                    return (a.performance?.quantizations?.q8?.ram_gb || 999) -
                        (b.performance?.quantizations?.q8?.ram_gb || 999);
                default:
                    return (b.aggregate_score || 0) - (a.aggregate_score || 0);
            }
        });
    }

    renderCards() {
        const grid = document.getElementById('model-grid');
        const noResults = document.getElementById('no-results');

        if (this.filteredData.length === 0) {
            grid.style.display = 'none';
            noResults.style.display = 'flex';
            return;
        }

        grid.style.display = 'grid';
        noResults.style.display = 'none';
        grid.innerHTML = '';

        this.filteredData.forEach((model, index) => {
            const card = this.createModelCard(model, index);
            grid.appendChild(card);
        });
    }

    createModelCard(model, index) {
        const card = document.createElement('div');
        card.className = 'model-card';
        if (index < 3) card.classList.add(`top-${index + 1}`);

        // Get best use cases
        const bestUseCases = this.getBestUseCases(model);
        const badges = bestUseCases.map(uc => `<span class="use-case-badge">${uc}</span>`).join('');

        // Get performance metrics
        const q8 = model.performance?.quantizations?.q8 || {};
        const q4 = model.performance?.quantizations?.q4 || {};

        // Get relevant score based on selected use case
        let primaryScore = model.aggregate_score || 0;
        let primaryLabel = 'Overall';
        if (this.selectedUseCase !== 'all' && model.use_cases?.[this.selectedUseCase]) {
            primaryScore = model.use_cases[this.selectedUseCase].score;
            primaryLabel = this.formatUseCaseName(this.selectedUseCase);
        }

        card.innerHTML = `
            <div class="card-header">
                <div class="card-rank">#${index + 1}</div>
                <div class="card-badges">${badges}</div>
            </div>
            
            <h3 class="card-title">${model.name}</h3>
            <p class="card-subtitle">${model.family} • ${model.parameters}</p>
            
            <div class="card-score">
                <div class="score-value">${primaryScore.toFixed(1)}</div>
                <div class="score-label">${primaryLabel}</div>
            </div>
            
            <div class="card-stats">
                <div class="stat">
                    <span class="stat-icon"></span>
                    <span class="stat-value">${q8.ram_gb || q4.ram_gb || '-'}GB</span>
                    <span class="stat-label">RAM (Q8)</span>
                </div>
                <div class="stat">
                    <span class="stat-icon">◇</span>
                    <span class="stat-value">${q4.tps_output || q8.tps_output || '-'}</span>
                    <span class="stat-label">TPS (Q4)</span>
                </div>
                <div class="stat">
                    <span class="stat-icon"></span>
                    <span class="stat-value">${this.formatContext(model.context_window)}</span>
                    <span class="stat-label">Context</span>
                </div>
            </div>

            <div class="card-use-cases">
                ${this.renderUseCaseBars(model)}
            </div>
            
            <div class="card-tags">
                ${(model.best_for || []).slice(0, 3).map(t => `<span class="tag">${t}</span>`).join('')}
            </div>

            <div class="card-actions">
                <a href="model.html?id=${model.id}" class="btn-primary">View Details</a>
                <a href="https://huggingface.co/${model.hf_repo}" target="_blank" class="btn-secondary"> HF</a>
            </div>
        `;

        return card;
    }

    getBestUseCases(model) {
        const useCases = [];
        if (model.use_cases) {
            Object.entries(model.use_cases).forEach(([key, value]) => {
                if (value.recommended) {
                    useCases.push(this.formatUseCaseName(key));
                }
            });
        }
        return useCases.slice(0, 2);
    }

    formatUseCaseName(key) {
        const names = {
            'rag': 'RAG',
            'function_calling': 'Function Call',
            'coding': 'Coding',
            'reasoning': 'Reasoning',
            'guardrails': 'Safety'
        };
        return names[key] || key;
    }

    formatContext(ctx) {
        if (!ctx) return '-';
        if (ctx >= 131072) return '128K';
        if (ctx >= 32768) return '32K';
        if (ctx >= 8192) return '8K';
        if (ctx >= 4096) return '4K';
        return `${(ctx / 1024).toFixed(0)}K`;
    }

    renderUseCaseBars(model) {
        const useCases = ['rag', 'function_calling', 'coding', 'reasoning', 'guardrails'];
        return useCases.map(uc => {
            const score = model.use_cases?.[uc]?.score || 0;
            const name = this.formatUseCaseName(uc);
            const isActive = this.selectedUseCase === uc;
            return `
                <div class="use-case-bar ${isActive ? 'active' : ''}">
                    <span class="bar-label">${name}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${score}%"></div>
                    </div>
                    <span class="bar-value">${score.toFixed(0)}</span>
                </div>
            `;
        }).join('');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new MarketplaceManager();
});
