class LeaderboardFilters {
    constructor(models, onFilterChange) {
        this.allModels = models;
        this.onFilterChange = onFilterChange;
        this.filters = this.loadFromURL() || {
            paramRange: [0, 10000000000],  // 0-10B default
            categories: [],
            minEfficiency: 0,
            minAccuracy: 0,
            maxEnergy: Infinity,
            sortBy: 'aggregate_score',
            sortOrder: 'desc',
            search: ''
        };

        this.initUI();
    }

    initUI() {
        // Attach listeners to UI elements if they exist
        const searchInput = document.getElementById('modelSearch');
        if (searchInput) {
            searchInput.value = this.filters.search;
            searchInput.addEventListener('input', (e) => {
                this.filters.search = e.target.value;
                this.applyFilters();
            });
        }

        // Add other listeners as needed (range sliders, checkboxes)
        // This assumes the HTML has these elements with specific IDs
    }

    loadFromURL() {
        const params = new URLSearchParams(window.location.search);
        if (!params.has('sort')) return null;

        return {
            paramRange: [
                parseInt(params.get('params_min')) || 0,
                parseInt(params.get('params_max')) || 10000000000
            ],
            categories: params.get('categories') ? params.get('categories').split(',') : [],
            minEfficiency: parseFloat(params.get('min_eff')) || 0,
            minAccuracy: parseFloat(params.get('min_acc')) || 0,
            maxEnergy: parseFloat(params.get('max_energy')) || Infinity,
            sortBy: params.get('sort') || 'aggregate_score',
            sortOrder: params.get('order') || 'desc',
            search: params.get('q') || ''
        };
    }

    saveToURL() {
        const params = new URLSearchParams();
        params.set('params_min', this.filters.paramRange[0]);
        params.set('params_max', this.filters.paramRange[1]);

        if (this.filters.categories.length > 0) {
            params.set('categories', this.filters.categories.join(','));
        }
        if (this.filters.minEfficiency > 0) {
            params.set('min_eff', this.filters.minEfficiency);
        }
        if (this.filters.minAccuracy > 0) {
            params.set('min_acc', this.filters.minAccuracy);
        }
        if (this.filters.maxEnergy < Infinity) {
            params.set('max_energy', this.filters.maxEnergy);
        }
        if (this.filters.search) {
            params.set('q', this.filters.search);
        }

        params.set('sort', this.filters.sortBy);
        params.set('order', this.filters.sortOrder);

        // Update URL without reload
        const newUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.replaceState({}, '', newUrl);
    }

    applyFilters() {
        let filtered = this.allModels.filter(m => {
            // Search
            if (this.filters.search && !m.name.toLowerCase().includes(this.filters.search.toLowerCase())) {
                return false;
            }

            // Parameter range (rough estimate parsing if string)
            let params = m.parameters;
            if (typeof params === 'string') {
                if (params.endsWith('B')) params = parseFloat(params) * 1e9;
                else if (params.endsWith('M')) params = parseFloat(params) * 1e6;
            }
            if (params < this.filters.paramRange[0] || params > this.filters.paramRange[1]) {
                return false;
            }

            // Category minimums
            for (const cat of this.filters.categories) {
                if (!m.scores || m.scores[cat] < 50) return false;
            }

            // Efficiency threshold
            if (this.filters.minEfficiency > 0 && (m.efficiency_score || 0) < this.filters.minEfficiency) {
                return false;
            }

            // Accuracy threshold
            if (this.filters.minAccuracy > 0 && (m.aggregate_score || 0) < this.filters.minAccuracy) {
                return false;
            }

            return true;
        });

        // Sort
        filtered.sort((a, b) => {
            const aVal = a[this.filters.sortBy] || 0;
            const bVal = b[this.filters.sortBy] || 0;

            return this.filters.sortOrder === 'desc'
                ? bVal - aVal
                : aVal - bVal;
        });

        this.saveToURL();

        if (this.onFilterChange) {
            this.onFilterChange(filtered);
        }
        return filtered;
    }
}
