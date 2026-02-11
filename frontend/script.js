/**
 * Main application logic for AI Research Assistant
 * Handles query submission, document search, and UI updates
 * Version: 3.0.0 - Added document search mode + persistent model optimizations
 */

class RAGAssistant {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.isProcessing = false;
        this.mode = 'ask'; // 'ask' or 'search'
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkHealth();
    }
    
    initializeElements() {
        // Input elements
        this.queryInput = document.getElementById('query-input');
        this.submitBtn = document.getElementById('submit-btn');
        this.submitBtnText = document.getElementById('submit-btn-text');
        
        // Mode toggle
        this.modeToggleInput = document.getElementById('mode-toggle-input');
        this.modeLabelAsk = document.getElementById('mode-label-ask');
        this.modeLabelSearch = document.getElementById('mode-label-search');
        this.exampleQueriesAsk = document.getElementById('example-queries-ask');
        this.exampleQueriesSearch = document.getElementById('example-queries-search');
        
        // Sections
        this.vizSection = document.getElementById('viz-section');
        this.resultsSection = document.getElementById('results-section');
        this.searchResultsSection = document.getElementById('search-results-section');
        this.inlineLoading = document.getElementById('inline-loading');
        this.loadingStatus = document.getElementById('loading-status');
        this.searchLoading = document.getElementById('search-loading');
        this.searchLoadingStatus = document.getElementById('search-loading-status');
        
        // Results elements (Ask mode)
        this.answerText = document.getElementById('answer-text');
        this.answerMetaModel = document.getElementById('answer-meta-model');
        this.answerMetaTime = document.getElementById('answer-meta-time');
        this.answerMetaWords = document.getElementById('answer-meta-words');
        this.sourcesContainer = document.getElementById('sources-container');
        
        // Results elements (Search mode)
        this.paperResultsContainer = document.getElementById('paper-results-container');
        this.searchResultsMeta = document.getElementById('search-results-meta');
        
        // Stage status elements
        this.stages = ['embedding', 'search', 'extraction', 'generation'];
        this.stageElements = {};
        
        this.stages.forEach(stage => {
            this.stageElements[stage] = {
                container: document.querySelector(`[data-stage="${stage}"]`),
                status: document.getElementById(`status-${stage}`),
                time: document.getElementById(`time-${stage}`)
            };
        });
    }
    
    attachEventListeners() {
        // Submit button
        this.submitBtn.addEventListener('click', () => this.handleSubmit());
        
        // Enter key in textarea (Shift+Enter for new line)
        this.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit();
            }
        });
        
        // Mode toggle
        this.modeToggleInput.addEventListener('change', () => this.handleModeChange());
        
        // Example queries (Ask mode)
        document.querySelectorAll('.example-query').forEach(btn => {
            btn.addEventListener('click', () => {
                if (this.mode !== 'ask') {
                    this.modeToggleInput.checked = false;
                    this.handleModeChange();
                }
                this.queryInput.value = btn.textContent;
                this.handleSubmit();
            });
        });
        
        // Example queries (Search mode)
        document.querySelectorAll('.example-query-search').forEach(btn => {
            btn.addEventListener('click', () => {
                if (this.mode !== 'search') {
                    this.modeToggleInput.checked = true;
                    this.handleModeChange();
                }
                this.queryInput.value = btn.textContent;
                this.handleSubmit();
            });
        });
    }
    
    handleModeChange() {
        this.mode = this.modeToggleInput.checked ? 'search' : 'ask';
        
        if (this.mode === 'search') {
            this.queryInput.placeholder = 'Search for papers about...';
            this.submitBtnText.textContent = 'Search';
            this.modeLabelSearch.classList.add('active');
            this.modeLabelAsk.classList.remove('active');
            this.exampleQueriesAsk.style.display = 'none';
            this.exampleQueriesSearch.style.display = 'flex';
        } else {
            this.queryInput.placeholder = 'Ask a question about machine learning research...';
            this.submitBtnText.textContent = 'Ask';
            this.modeLabelAsk.classList.add('active');
            this.modeLabelSearch.classList.remove('active');
            this.exampleQueriesAsk.style.display = 'flex';
            this.exampleQueriesSearch.style.display = 'none';
        }
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            const paperCount = document.getElementById('paper-count');
            const embeddingCount = document.getElementById('embedding-count');
            
            if (paperCount && data.paper_count) {
                paperCount.textContent = this.formatNumber(data.paper_count);
            }
            
            if (embeddingCount && data.vector_store_count) {
                embeddingCount.textContent = this.formatNumber(data.vector_store_count);
            }
        } catch (error) {
            console.error('Health check failed:', error);
        }
        
        // Set initial mode label state
        this.modeLabelAsk.classList.add('active');
    }
    
    async handleSubmit() {
        const query = this.queryInput.value.trim();
        
        if (!query || this.isProcessing) {
            return;
        }
        
        this.isProcessing = true;
        this.submitBtn.disabled = true;
        
        if (this.mode === 'ask') {
            await this.handleAskSubmit(query);
        } else {
            await this.handleSearchSubmit(query);
        }
        
        this.isProcessing = false;
        this.submitBtn.disabled = false;
    }
    
    // ==================== ASK MODE ====================
    
    async handleAskSubmit(query) {
        // Show results section with inline loading
        this.resultsSection.style.display = 'block';
        this.searchResultsSection.style.display = 'none';
        this.inlineLoading.style.display = 'flex';
        this.answerText.style.display = 'none';
        this.loadingStatus.textContent = 'Initializing...';
        
        // Clear previous
        this.answerText.innerHTML = '';
        this.sourcesContainer.innerHTML = '';
        this.answerMetaModel.textContent = '';
        this.answerMetaTime.textContent = '';
        this.answerMetaWords.textContent = '';
        
        // Show pipeline viz
        this.vizSection.style.display = 'block';
        this.resetStages();
        
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        try {
            await this.streamQuery(query);
        } catch (error) {
            console.error('Query error:', error);
            this.inlineLoading.style.display = 'none';
            this.answerText.style.display = 'block';
            this.answerText.innerHTML = '<p style="color: var(--error);">Error: ' + this.escapeHtml(error.message) + '</p>';
        }
    }
    
    async streamQuery(query) {
        const eventSource = new EventSource(
            this.apiBaseUrl + '/query/stream?query=' + encodeURIComponent(query) + '&k=6'
        );
        
        return new Promise((resolve, reject) => {
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                switch (data.stage) {
                    case 'start':
                        this.loadingStatus.textContent = 'Starting pipeline...';
                        break;
                    
                    case 'search':
                        if (data.status === 'running') {
                            this.updateStage('embedding', 'running');
                            this.updateStage('search', 'running');
                            this.loadingStatus.textContent = 'Embedding query and searching vector store...';
                        } else if (data.status === 'complete') {
                            this.updateStage('search', 'complete', data.time);
                            this.loadingStatus.textContent = 'Found ' + data.results + ' relevant sources';
                        }
                        break;
                    
                    case 'extraction':
                        if (data.status === 'running') {
                            this.updateStage('extraction', 'running');
                            this.loadingStatus.textContent = 'Extracting relevant context from papers...';
                        } else if (data.status === 'complete') {
                            this.updateStage('extraction', 'complete', data.time);
                            this.loadingStatus.textContent = 'Context extracted successfully';
                        }
                        break;
                    
                    case 'generation':
                        if (data.status === 'running') {
                            this.updateStage('generation', 'running');
                            this.loadingStatus.textContent = 'Generating detailed answer...';
                        } else if (data.status === 'complete') {
                            this.updateStage('generation', 'complete', data.time);
                        }
                        break;
                    
                    case 'complete':
                        this.displayResults(data.result);
                        eventSource.close();
                        resolve();
                        break;
                    
                    case 'error':
                        eventSource.close();
                        reject(new Error(data.message));
                        break;
                }
            };
            
            eventSource.onerror = function() {
                eventSource.close();
                reject(new Error('Connection to server lost'));
            };
        });
    }
    
    updateStage(stage, status, time) {
        const elements = this.stageElements[stage];
        if (!elements) return;
        
        elements.container.setAttribute('data-status', status);
        elements.status.textContent = status;
        elements.status.className = 'stage-status ' + status;
        
        if (time !== undefined && time !== null) {
            elements.time.textContent = time.toFixed(1) + 's';
        }
    }
    
    resetStages() {
        this.stages.forEach(stage => {
            const elements = this.stageElements[stage];
            if (!elements) return;
            
            elements.container.setAttribute('data-status', '');
            elements.status.textContent = 'Waiting';
            elements.status.className = 'stage-status';
            elements.time.textContent = '';
        });
    }
    
    displayResults(result) {
        this.inlineLoading.style.display = 'none';
        this.answerText.style.display = 'block';
        
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        // Display answer
        this.answerText.innerHTML = this.formatAnswer(result.answer);
        
        // Display metadata
        this.answerMetaModel.textContent = 'Model: ' + result.metadata.model;
        this.answerMetaTime.textContent = 'Time: ' + result.timing.total.toFixed(1) + 's';
        this.answerMetaWords.textContent = result.metadata.answer_word_count + ' words';
        
        // Display sources
        this.sourcesContainer.innerHTML = '';
        if (result.sources && result.sources.length > 0) {
            result.sources.forEach(source => {
                this.sourcesContainer.appendChild(this.createSourceCard(source));
            });
        }
    }
    
    // ==================== SEARCH MODE ====================
    
    async handleSearchSubmit(query) {
        // Show search results section with loading
        this.resultsSection.style.display = 'none';
        this.vizSection.style.display = 'none';
        this.searchResultsSection.style.display = 'block';
        this.searchLoading.style.display = 'flex';
        this.searchLoadingStatus.textContent = 'Searching papers...';
        this.paperResultsContainer.innerHTML = '';
        this.searchResultsMeta.textContent = '';
        
        this.searchResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        try {
            await this.streamSearch(query);
        } catch (error) {
            console.error('Search error:', error);
            this.searchLoading.style.display = 'none';
            this.paperResultsContainer.innerHTML = '<p style="color: var(--error);">Error: ' + this.escapeHtml(error.message) + '</p>';
        }
    }
    
    async streamSearch(query) {
        const eventSource = new EventSource(
            this.apiBaseUrl + '/search/papers?query=' + encodeURIComponent(query) + '&k=9'
        );
        
        return new Promise((resolve, reject) => {
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                switch (data.stage) {
                    case 'start':
                        this.searchLoadingStatus.textContent = 'Searching papers...';
                        break;
                    
                    case 'search':
                        if (data.status === 'complete') {
                            this.searchLoadingStatus.textContent = 'Found ' + data.results + ' papers, generating summaries...';
                        }
                        break;
                    
                    case 'summarising':
                        if (data.status === 'running') {
                            this.searchLoadingStatus.textContent = 'Summarising ' + data.total + ' papers...';
                        }
                        break;
                    
                    case 'paper':
                        // A paper result arrived — add it to the grid progressively
                        this.searchLoading.style.display = 'none';
                        this.paperResultsContainer.appendChild(this.createPaperCard(data.paper));
                        this.searchResultsMeta.textContent = data.progress + ' of ' + data.total + ' papers';
                        break;
                    
                    case 'complete':
                        this.searchLoading.style.display = 'none';
                        var total = data.papers ? data.papers.length : 0;
                        this.searchResultsMeta.textContent = total + ' papers found in ' + data.timing.total.toFixed(1) + 's';
                        eventSource.close();
                        resolve();
                        break;
                    
                    case 'error':
                        eventSource.close();
                        reject(new Error(data.message));
                        break;
                }
            };
            
            eventSource.onerror = function() {
                eventSource.close();
                reject(new Error('Connection to server lost'));
            };
        });
    }
    
    createPaperCard(paper) {
        const card = document.createElement('div');
        card.className = 'paper-card';
        
        // Relevance badge
        const relevancePct = (paper.relevance * 100).toFixed(0);
        
        // Authors formatting
        var authorsStr = '';
        if (paper.authors && paper.authors.length > 0) {
            if (paper.authors.length <= 3) {
                authorsStr = paper.authors.join(', ');
            } else {
                authorsStr = paper.authors.slice(0, 3).join(', ') + ' +' + (paper.authors.length - 3) + ' more';
            }
        }
        
        // Categories
        var categoriesHtml = '';
        if (paper.categories && paper.categories.length > 0) {
            categoriesHtml = paper.categories
                .map(c => '<span class="paper-category">' + this.escapeHtml(c) + '</span>')
                .join('');
        }
        
        // Published date formatting
        var dateStr = '';
        if (paper.published) {
            var d = new Date(paper.published);
            if (!isNaN(d.getTime())) {
                dateStr = d.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: 'numeric' });
            } else {
                dateStr = paper.published;
            }
        }
        
        var metaHtml = '<span class="paper-id">arXiv: ' + this.escapeHtml(paper.arxiv_id) + '</span>';
        if (dateStr) {
            metaHtml += '<span class="paper-date">' + dateStr + '</span>';
        }
        if (paper.word_count) {
            metaHtml += '<span class="paper-words">' + this.formatNumber(paper.word_count) + ' words</span>';
        }
        
        card.innerHTML = '<div class="paper-card-header">' +
            '<span class="paper-number">' + paper.number + '</span>' +
            '<span class="paper-relevance">' + relevancePct + '% match</span>' +
            '</div>' +
            '<h3 class="paper-title">' + this.escapeHtml(paper.title || 'Untitled Paper') + '</h3>' +
            (authorsStr ? '<div class="paper-authors">' + this.escapeHtml(authorsStr) + '</div>' : '') +
            '<div class="paper-meta">' + metaHtml + '</div>' +
            (categoriesHtml ? '<div class="paper-categories">' + categoriesHtml + '</div>' : '') +
            '<p class="paper-summary">' + this.escapeHtml(paper.summary) + '</p>';
        
        return card;
    }
    
    // ==================== SHARED UTILITIES ====================
    
    formatAnswer(text) {
        // Normalize consecutive citations: [1][2][3] → [1, 2, 3]
        text = text.replace(/\](\s*)\[/g, function(match, space) {
            return ', ';
        });
        // Fix resulting format: [1, 2, 3] (the above turns [1][2] into [1, 2])
        // Handle edge case where first ] was removed
        text = text.replace(/\[(\d+(?:,\s*\d+)*),\s*(\d+(?:,\s*\d+)*)\]/g, function(match) {
            return match; // already correct
        });

        var paragraphs = text.split('\n\n');
        var formatted = '';
        
        for (var i = 0; i < paragraphs.length; i++) {
            var para = paragraphs[i].trim();
            if (!para) continue;
            
            if (para.indexOf('\u2022') !== -1 || para.indexOf('* ') !== -1) {
                var items = para.split(/\n/).filter(function(line) { return line.trim(); });
                formatted += '<ul>';
                for (var j = 0; j < items.length; j++) {
                    var item = items[j].replace(/^[\u2022*]\s*/, '').trim();
                    if (item) {
                        formatted += '<li>' + this.formatCitations(this.escapeHtml(item)) + '</li>';
                    }
                }
                formatted += '</ul>';
            } else {
                formatted += '<p>' + this.formatCitations(this.escapeHtml(para)) + '</p>';
            }
        }
        
        return formatted;
    }
    
    formatCitations(html) {
        // Style citation references as inline badges: [1], [1, 2], etc.
        return html.replace(/\[(\d+(?:,\s*\d+)*)\]/g, function(match, nums) {
            return '<span class="citation-ref">[' + nums + ']</span>';
        });
    }
    
    escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    createSourceCard(source) {
        var card = document.createElement('div');
        card.className = 'source-card' + (source.cited ? ' cited' : '');
        
        var header = document.createElement('div');
        header.className = 'source-header';
        
        var number = document.createElement('div');
        number.className = 'source-number';
        number.textContent = source.number;
        header.appendChild(number);
        
        if (source.cited) {
            var citedBadge = document.createElement('div');
            citedBadge.className = 'source-cited';
            citedBadge.textContent = 'Cited';
            header.appendChild(citedBadge);
        }
        
        var relevanceBadge = document.createElement('div');
        relevanceBadge.className = 'source-relevance';
        relevanceBadge.textContent = (source.relevance * 100).toFixed(0) + '% match';
        header.appendChild(relevanceBadge);
        
        card.appendChild(header);
        
        if (source.title && source.title !== 'Unknown') {
            var title = document.createElement('div');
            title.className = 'source-title';
            title.textContent = source.title;
            card.appendChild(title);
        }
        
        if (source.authors) {
            var authors = document.createElement('div');
            authors.className = 'source-authors';
            authors.textContent = source.authors;
            card.appendChild(authors);
        }
        
        var meta = document.createElement('div');
        meta.className = 'source-meta';
        meta.innerHTML = '<span>arXiv: ' + this.escapeHtml(source.arxiv_id) + '</span>' +
            '<span>Section: ' + this.escapeHtml(source.section) + '</span>';
        if (source.categories) {
            meta.innerHTML += '<span>' + this.escapeHtml(source.categories) + '</span>';
        }
        card.appendChild(meta);
        
        return card;
    }
    
    showError(message) {
        alert('Error: ' + message);
    }
    
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(0) + 'K';
        }
        return num.toString();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.ragAssistant = new RAGAssistant();
});