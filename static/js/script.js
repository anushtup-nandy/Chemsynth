// Global variables
let synthesisRoutesData = [];
let currentTargetSMILES = ''; // Store the resolved SMILES for the "New Route" feature

document.addEventListener('DOMContentLoaded', function() {
    console.log('ChemSynthAI Platform Initialized');

    // --- Element Selectors ---
    const searchButton = document.getElementById('literature-search-btn');
    const targetMoleculeInput = document.getElementById('target-molecule-input');
    const searchTermsInput = document.getElementById('search-terms-input');
    const literatureResultsContainer = document.getElementById('literature-results-container');
    const literatureResultsCount = document.getElementById('literature-results-count');
    const synthesisSection = document.getElementById('synthesis-route-section');
    const routeTabsContainer = document.getElementById('synthesis-route-tabs-container');
    const routeContentContainer = document.getElementById('synthesis-route-content-container');
    const newRouteButton = document.getElementById('new-route-btn'); // Select the static button

    // --- Event Listeners ---
    if (searchButton) {
        searchButton.addEventListener('click', handleSearch);
    }
    if (newRouteButton) {
        newRouteButton.addEventListener('click', handleNewRoute);
    }

    // --- Main Handler Functions ---
    function handleSearch(event) {
        event.preventDefault();
        const identifier = targetMoleculeInput.value.trim();
        const keywords = searchTermsInput.value.trim();
        const literatureQuery = (identifier + ' ' + keywords).trim();

        if (!identifier) {
            alert("Please enter a target molecule identifier (SMILES, Name, or CAS).");
            return;
        }

        if (literatureQuery) {
            fetchLiterature(literatureQuery);
        }
        fetchSynthesisPlan(identifier);
    }
    
    async function handleNewRoute() {
        if (!currentTargetSMILES) {
            alert("Please perform a search for a target molecule first.");
            return;
        }

        const suggestion = prompt("Suggest an alternative reaction or approach (e.g., 'Use a Suzuki coupling for step 1'):");
        if (!suggestion || suggestion.trim() === '') {
            return; // User cancelled or entered nothing
        }

        const originalButtonText = this.innerHTML;
        this.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Generating...';
        this.disabled = true;

        try {
            const response = await fetch('/api/generate_new_route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    suggestion: suggestion,
                    existing_routes: synthesisRoutesData,
                    target_smiles: currentTargetSMILES
                }),
            });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate new route.');
            }
            
            addNewRouteToDisplay(data.new_route);

        } catch (error) {
            console.error('Error generating new route:', error);
            alert(`Error: ${error.message}`);
        } finally {
            this.innerHTML = originalButtonText;
            this.disabled = false;
        }
    }
    
    function addNewRouteToDisplay(newRoute) {
        newRoute.id = `route_hypothetical_${synthesisRoutesData.length}`;
        synthesisRoutesData.push(newRoute);
        
        renderRouteTabs(synthesisRoutesData);
        renderSingleRoute(newRoute);

        document.querySelectorAll('.route-tab').forEach(tab => {
            tab.classList.remove('border-blue-500', 'text-blue-500');
            tab.classList.add('border-transparent', 'text-gray-500');
        });
        const newTabEl = routeTabsContainer.querySelector(`[data-route-id="${newRoute.id}"]`);
        if(newTabEl) {
            newTabEl.classList.add('border-blue-500', 'text-blue-500');
            newTabEl.classList.remove('border-transparent', 'text-gray-500');
        }
    }

    // --- API Fetching Functions ---
    async function fetchSynthesisPlan(identifier) {
        synthesisSection.classList.remove('hidden');
        routeTabsContainer.innerHTML = '';
        routeContentContainer.innerHTML = '<p class="text-center text-gray-400 py-8"><i class="fas fa-spinner fa-spin mr-2"></i>Resolving identifier and planning routes...</p>';
        
        try {
            const response = await fetch('/api/plan_synthesis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier: identifier }),
            });
            
            const data = await response.json();
            if (!response.ok) { throw new Error(data.error || "An unknown error occurred."); }
            
            if (data.routes && data.routes.length > 0) {
                synthesisRoutesData = data.routes;
                if (data.routes[0].steps.length > 0) {
                    currentTargetSMILES = data.routes[0].steps[data.routes[0].steps.length - 1].product.smiles;
                }
                renderRouteTabs(synthesisRoutesData);
                renderSingleRoute(synthesisRoutesData[0]);
            } else {
                routeContentContainer.innerHTML = '<p class="text-center text-yellow-400 py-8">Could not find any synthesis routes for the target molecule.</p>';
            }
        } catch (error) {
            console.error('Error fetching synthesis plan:', error);
            routeContentContainer.innerHTML = `<div class="text-center py-8"><p class="text-red-400 mb-2">Failed to plan synthesis.</p><p class="text-gray-500 text-sm">${error.message}</p></div>`;
        }
    }

    async function fetchLiterature(query) {
        literatureResultsContainer.innerHTML = '<p class="text-center text-gray-400"><i class="fas fa-spinner fa-spin mr-2"></i>Searching literature...</p>';
        literatureResultsCount.textContent = 'Searching...';
        try {
            const response = await fetch('/api/literature_search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, max_results: 5 }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            renderLiteratureResults(data.papers);
        } catch (error) {
            console.error('Error fetching literature results:', error);
            literatureResultsContainer.innerHTML = `<p class="text-center text-red-400">Failed to load literature: ${error.message}</p>`;
            literatureResultsCount.textContent = 'Literature Results (Error)';
        }
    }

    // --- Rendering Functions ---
    function renderRouteTabs(routes) {
        let tabsHtml = `<nav class="-mb-px flex space-x-8">`;
        routes.forEach((route, index) => {
            const defaultName = `Route ${String.fromCharCode(65 + index)}`;
            const routeName = route.name || defaultName; 
            const isActive = index === (routes.length - 1) ? 'border-blue-500 text-blue-500' : 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300';
            tabsHtml += `<a href="#" class="${isActive} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm route-tab" data-route-id="${route.id}">
                            ${routeName} (${route.overall_yield.toFixed(1)}% yield)
                        </a>`;
        });
        tabsHtml += '</nav>';
        routeTabsContainer.innerHTML = tabsHtml;

        document.querySelectorAll('.route-tab').forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                const routeId = this.dataset.routeId;
                const routeData = synthesisRoutesData.find(r => r.id === routeId);
                
                document.querySelectorAll('.route-tab').forEach(t => {
                    t.classList.remove('border-blue-500', 'text-blue-500');
                    t.classList.add('border-transparent', 'text-gray-500');
                });
                this.classList.add('border-blue-500', 'text-blue-500');
                this.classList.remove('border-transparent', 'text-gray-500');

                renderSingleRoute(routeData);
            });
        });
    }

    function renderSingleRoute(routeData) {
        let contentHtml = `
            ${renderRouteVisualization(routeData)}
            ${renderRouteDetails(routeData.steps)}
            ${renderRouteEvaluation(routeData.evaluation)}
        `;
        routeContentContainer.innerHTML = contentHtml;
    }

    function renderRouteVisualization(route) {
        let vizHtml = '<div class="reaction-visualization mb-6"><div class="flex flex-col md:flex-row items-center justify-center mb-8 overflow-x-auto p-4">';
        if (!route.steps || route.steps.length === 0) return '';
        
        const firstReactants = route.steps[0].reactants;
        const reactantNames = firstReactants.map(r => r.formula || 'Reactant').join(' + ');
        vizHtml += `
            <div class="molecule-display flex-shrink-0">
                <div class="text-center p-4">
                    <div class="text-xs text-gray-400 mb-1">Starting Material(s)</div>
                    <div class="font-bold">${reactantNames}</div>
                </div>
            </div>
        `;

        route.steps.forEach(step => {
            const isFinalProduct = step.step_number === route.steps.length;
            const borderColor = isFinalProduct ? 'border-2 border-blue-500' : '';
            const textColor = isFinalProduct ? 'text-blue-400' : '';
            const titleText = isFinalProduct ? 'Target Molecule' : `Intermediate`;
            vizHtml += `
                <div class="reaction-arrow flex-shrink-0 mx-2 my-4 md:my-0"><i class="fas fa-long-arrow-alt-right text-2xl"></i></div>
                <div class="molecule-display ${borderColor} flex-shrink-0">
                    <div class="text-center p-4">
                        <div class="text-xs text-gray-400 mb-1">Step ${step.step_number}: ${step.yield.toFixed(1)}% yield</div>
                        <div class="font-bold ${textColor}">${titleText}</div>
                        <div class="text-xs text-gray-400 mt-1">${step.product.formula}</div>
                    </div>
                </div>
            `;
        });
        vizHtml += '</div></div>';
        return vizHtml;
    }

    function renderRouteDetails(steps) {
        let detailsHtml = '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">';
        if (!steps) return '';
        steps.forEach(step => {
            detailsHtml += `
                <div class="bg-gray-700 rounded-lg p-4">
                    <h4 class="font-bold mb-2">Step ${step.step_number}: ${step.title}</h4>
                    <p class="text-sm text-gray-300 mb-3">${step.reagents_conditions}</p>
                    <div class="text-xs text-gray-400 mb-2"><span class="font-medium">Predicted Yield:</span> ${step.yield.toFixed(1)}%</div>
                    <div class="text-xs text-gray-400"><span class="font-medium">Notes:</span> ${step.source_notes}</div>
                </div>
            `;
        });
        detailsHtml += '</div>';
        return detailsHtml;
    }

    function renderRouteEvaluation(evaluation) {
        if (!evaluation) return '';
        const advantages = (evaluation.advantages || []).map(adv => `<li class="flex items-start"><i class="fas fa-check-circle text-green-500 mt-1 mr-2 flex-shrink-0"></i><span>${adv}</span></li>`).join('');
        const challenges = (evaluation.challenges || []).map(chal => `<li class="flex items-start"><i class="fas fa-exclamation-triangle text-yellow-500 mt-1 mr-2 flex-shrink-0"></i><span>${chal}</span></li>`).join('');
        return `
            <div class="bg-gray-700 rounded-lg p-6">
                <h3 class="font-bold mb-4">Route Evaluation</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h4 class="text-sm font-medium text-gray-300 mb-2">Advantages</h4>
                        <ul class="text-sm text-gray-400 space-y-2">${advantages || '<li>N/A</li>'}</ul>
                    </div>
                    <div>
                        <h4 class="text-sm font-medium text-gray-300 mb-2">Challenges</h4>
                        <ul class="text-sm text-gray-400 space-y-2">${challenges || '<li>N/A</li>'}</ul>
                    </div>
                </div>
            </div>
        `;
    }

    function renderLiteratureResults(papers) {
        if (!papers || papers.length === 0) {
            literatureResultsContainer.innerHTML = '<p class="text-center text-gray-400">No relevant papers found.</p>';
            literatureResultsCount.textContent = 'Literature Results (0 papers found)';
            return;
        }
        
        literatureResultsCount.textContent = `Literature Results (${papers.length} papers found)`;
        
        let html = '';
        papers.forEach(paper => {
            const keywordsHtml = (paper.keywords || []).map(kw => 
                `<span class="text-xs bg-gray-600 text-gray-300 px-2 py-1 rounded">${kw}</span>`
            ).join(' ');
            
            const renderedAbstract = marked.parse(paper.abstract || 'No abstract available.');

            html += `
                <div class="bg-gray-700 rounded-lg p-4 mb-4">
                    <div class="flex justify-between items-start">
                        <div>
                            <h4 class="font-bold text-blue-400 hover:text-blue-300 cursor-pointer">${paper.title}</h4>
                            <p class="text-sm text-gray-400 mb-2">${paper.source}</p>
                        </div>
                        <div class="flex items-center space-x-2 flex-shrink-0 ml-4">
                            <span class="text-xs bg-blue-900 text-blue-300 px-2 py-1 rounded">Category: ${paper.impact}</span>
                        </div>
                    </div>
                    <div class="text-sm text-gray-300 mb-3">${renderedAbstract}</div>
                    <div class="flex flex-wrap gap-2">${keywordsHtml}</div>
                </div>
            `;
        });
        literatureResultsContainer.innerHTML = html;
    }
});