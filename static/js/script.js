// Global variables
let synthesisRoutesData = [];
let literatureResults = [];
let sourcingData = {}; // <<< NEW: Store sourcing/cost analysis results by route ID
let currentTargetSMILES = ''; // Store the resolved SMILES for the "New Route" feature

document.addEventListener('DOMContentLoaded', function() {
    console.log('ChemSynthAI Platform Initialized');

    // --- Element Selectors ---
    const searchButton = document.getElementById('literature-search-btn');
    const targetMoleculeInput = document.getElementById('target-molecule-input');
    const searchTermsInput = document.getElementById('search-terms-input');
    const literatureResultsContainer = document.getElementById('literature-results-container');
    const literatureResultsCount = document.getElementById('literature-results-count');
    const literatureSortSelect = document.getElementById('literature-sort-select');
    
    const synthesisContent = document.getElementById('synthesis-content');
    const sourcingContent = document.getElementById('sourcing-content');
    const costContent = document.getElementById('cost-content');
    const routeTabsContainer = document.getElementById('synthesis-route-tabs-container');
    const routeContentContainer = document.getElementById('synthesis-route-content-container');
    const newRouteButton = document.getElementById('new-route-btn');

    const progressBar = document.getElementById('project-progress-bar');
    const progressStepsContainer = document.getElementById('project-progress-steps');
    const mainTabs = document.getElementById('main-tabs');

    // --- Initial State Setup ---
    updateProgress(0); // Initialize progress bar
    setupTabControls(); // Activate main tabs

    // --- Event Listeners ---
    searchButton.addEventListener('click', handleSearch);
    newRouteButton.addEventListener('click', handleNewRoute);
    literatureSortSelect.addEventListener('change', handleSortLiterature);

    // --- Main Handler Functions ---
    async function handleSearch(event) {
        event.preventDefault();
        const identifier = targetMoleculeInput.value.trim();
        const keywords = searchTermsInput.value.trim();

        if (!identifier) {
            alert("Please enter a target molecule identifier (SMILES, Name, or CAS).");
            return;
        }

        updateProgress(0, true); // Reset and show loading
        
        try {
            const resolvedData = await resolveIdentifier(identifier);
            if (resolvedData) {
                const literatureQuery = (resolvedData.name + ' ' + keywords).trim();
                fetchLiterature(literatureQuery);
                fetchSynthesisPlan(identifier); 
                document.getElementById('project-target-molecule-name').textContent = `Target: ${resolvedData.name}`;
                document.getElementById('project-target-molecule-formula').textContent = resolvedData.smiles;
            }
        } catch (error) {
            alert(error.message);
            updateProgress(0); 
        }
    }

    async function handleNewRoute() {
        if (!currentTargetSMILES) {
            alert("Please perform a search for a target molecule first.");
            return;
        }

        const suggestion = prompt("Suggest an alternative reaction or approach (e.g., 'Use a Suzuki coupling for step 1'):");
        if (!suggestion || suggestion.trim() === '') return;

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
            if (!response.ok) throw new Error(data.error || 'Failed to generate new route.');
            
            addNewRouteToDisplay(data.new_route);
        } catch (error) {
            console.error('Error generating new route:', error);
            alert(`Error: ${error.message}`);
        } finally {
            this.innerHTML = '<i class="fas fa-plus mr-2"></i> New Route';
            this.disabled = false;
        }
    }

    function handleSortLiterature() {
        const sortBy = literatureSortSelect.value;
        let sortedResults = [...literatureResults];

        if (sortBy === 'date') {
            sortedResults.sort((a, b) => new Date(b.published) - new Date(a.published));
        } else if (sortBy === 'citations') {
            sortedResults.sort((a, b) => {
                const versionA = parseInt(a.entry_id.slice(-1)) || 0;
                const versionB = parseInt(b.entry_id.slice(-1)) || 0;
                return versionB - versionA;
            });
        }
        else {
             sortedResults = [...literatureResults];
        }
        
        renderLiteratureResults(sortedResults);
    }

    function addNewRouteToDisplay(newRoute) {
        newRoute.id = newRoute.id || `route_hypothetical_${synthesisRoutesData.length}`;
        synthesisRoutesData.push(newRoute);
        fetchSourcingAndCost([newRoute]); // Fetch analysis for the new route
        renderRouteTabs(synthesisRoutesData, newRoute.id);
        renderSingleRoute(newRoute);
    }

    // --- API Fetching Functions ---
    async function resolveIdentifier(identifier) {
        const response = await fetch('/api/resolve_identifier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to resolve molecule identifier.');
        }
        return data;
    }

    async function fetchSynthesisPlan(identifier) {
        synthesisContent.classList.remove('hidden');
        routeTabsContainer.innerHTML = '';
        routeContentContainer.innerHTML = '<p class="text-center text-gray-400 py-8"><i class="fas fa-spinner fa-spin mr-2"></i>Resolving identifier and planning routes...</p>';
        
        try {
            const response = await fetch('/api/plan_synthesis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier: identifier }),
            });
            
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "An unknown error occurred.");
            
            if (data.routes && data.routes.length > 0) {
                synthesisRoutesData = data.routes;
                if (data.routes[0].steps.length > 0) {
                    currentTargetSMILES = data.routes[0].steps[data.routes[0].steps.length - 1].product.smiles;
                }
                renderRouteTabs(synthesisRoutesData, synthesisRoutesData[0].id);
                renderSingleRoute(synthesisRoutesData[0]);
                updateProgress(2); // Synthesis complete
                fetchSourcingAndCost(synthesisRoutesData); // <<< NEW: Fetch analysis after routes are found
            } else {
                routeContentContainer.innerHTML = '<p class="text-center text-yellow-400 py-8">Could not find any synthesis routes for the target molecule.</p>';
                updateProgress(1);
            }
        } catch (error) {
            console.error('Error fetching synthesis plan:', error);
            routeContentContainer.innerHTML = `<div class="text-center py-8"><p class="text-red-400 mb-2">Failed to plan synthesis.</p><p class="text-gray-500 text-sm">${error.message}</p></div>`;
            updateProgress(1);
        }
    }

    async function fetchLiterature(query) {
        literatureResultsContainer.innerHTML = '<p class="text-center text-gray-400"><i class="fas fa-spinner fa-spin mr-2"></i>Searching literature...</p>';
        literatureResultsCount.textContent = 'Searching...';
        try {
            const response = await fetch('/api/literature_search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, max_results: 10 }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            literatureResults = data.papers || [];
            renderLiteratureResults(literatureResults);
            updateProgress(1);
        } catch (error) {
            console.error('Error fetching literature results:', error);
            literatureResultsContainer.innerHTML = `<p class="text-center text-red-400">Failed to load literature: ${error.message}</p>`;
            literatureResultsCount.textContent = 'Literature Results (Error)';
            updateProgress(0);
        }
    }

    // <<< NEW FUNCTION: Fetches sourcing and cost analysis for routes >>>
    async function fetchSourcingAndCost(routes) {
        updateProgress(3, true); // Sourcing step is loading
        const analysisPromises = routes.map(route =>
            fetch('/api/analyze_sourcing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ route_steps: route.steps, target_amount_g: 1.0 })
            })
            .then(res => res.ok ? res.json() : Promise.reject('Failed to fetch analysis'))
            .then(data => ({ routeId: route.id, analysis: data }))
            .catch(error => ({ routeId: route.id, error }))
        );

        const results = await Promise.all(analysisPromises);
        results.forEach(result => {
            if (result.analysis) {
                sourcingData[result.routeId] = result.analysis;
            } else {
                sourcingData[result.routeId] = { error: "Failed to load analysis for this route." };
            }
        });
        
        updateProgress(4); // Sourcing & Costing complete
    }
    
    // --- UI Control and Rendering ---
    function setupTabControls() {
        mainTabs.addEventListener('click', (e) => {
            e.preventDefault();
            const clickedTab = e.target.closest('.tab-link');
            if (!clickedTab) return;

            const tabName = clickedTab.dataset.tab;

            document.querySelectorAll('.tab-link').forEach(tab => {
                tab.classList.remove('border-blue-500', 'text-blue-500');
                tab.classList.add('border-transparent', 'text-gray-500');
            });
            clickedTab.classList.add('border-blue-500', 'text-blue-500');
            clickedTab.classList.remove('border-transparent', 'text-gray-500');

            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.add('hidden');
            });
            const activeContent = document.getElementById(`${tabName}-content`);
            activeContent.classList.remove('hidden');

            // <<< NEW: Render content for sourcing/cost tabs when they are selected >>>
            if (tabName === 'sourcing' || tabName === 'cost') {
                const activeRouteTab = document.querySelector('.route-tab.border-blue-500');
                if (activeRouteTab) {
                    const activeRouteId = activeRouteTab.dataset.routeId;
                    if (tabName === 'sourcing') renderSourcingInfo(activeRouteId);
                    else if (tabName === 'cost') renderCostAnalysis(activeRouteId);
                }
            }
        });
    }
    
    function updateProgress(stage, isLoading = false) {
        const stages = ['Literature', 'Synthesis', 'Sourcing', 'Costing', 'Knowledge'];
        let width = stage * 20;
        progressBar.style.width = `${width}%`;

        let stepsHtml = '';
        stages.forEach((name, index) => {
            let statusClass = 'text-gray-500'; 
            if (index < stage) {
                statusClass = 'text-green-400';
            } else if (index === stage) {
                statusClass = isLoading ? 'text-blue-400 animate-pulse' : 'text-blue-400';
            }
            stepsHtml += `<span class="${statusClass}">${name}</span>`;
        });
        progressStepsContainer.innerHTML = stepsHtml;
    }

    function renderRouteTabs(routes, activeRouteId) {
        let tabsHtml = `<nav class="-mb-px flex space-x-8">`;
        routes.forEach((route, index) => {
            const defaultName = `Route ${String.fromCharCode(65 + index)}`;
            const routeName = route.name || defaultName;
            const isActive = route.id === activeRouteId ? 'border-blue-500 text-blue-500' : 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300';
            tabsHtml += `<a href="#" class="${isActive} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm route-tab" data-route-id="${route.id}">
                            ${routeName} (${(route.overall_yield || 0).toFixed(1)}% yield)
                        </a>`;
        });
        tabsHtml += '</nav>';
        routeTabsContainer.innerHTML = tabsHtml;

        document.querySelectorAll('.route-tab').forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                const routeId = this.dataset.routeId;
                const routeData = synthesisRoutesData.find(r => r.id === routeId);
                renderRouteTabs(synthesisRoutesData, routeId);
                renderSingleRoute(routeData);

                // <<< NEW: Update sourcing/cost view if those tabs are active >>>
                const activeMainTab = document.querySelector('#main-tabs .tab-link.border-blue-500');
                if (activeMainTab) {
                    const tabName = activeMainTab.dataset.tab;
                    if (tabName === 'sourcing') renderSourcingInfo(routeId);
                    if (tabName === 'cost') renderCostAnalysis(routeId);
                }
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
        let vizHtml = '<div class="reaction-visualization mb-6"><div class="flex items-center justify-start mb-8 overflow-x-auto p-4">';
        if (!route.steps || route.steps.length === 0) return '';
        
        const firstReactants = route.steps[0].reactants;
        const reactantNames = firstReactants.map(r => r.formula || 'Reactant').join(' + ');
        vizHtml += `
            <div class="molecule-display flex-shrink-0">
                <div class="text-center p-4">
                    <div class="text-xs text-gray-400 mb-1">Starting Material(s)</div>
                    <div class="font-bold">${reactantNames}</div>
                </div>
            </div>`;

        route.steps.forEach(step => {
            const isFinalProduct = step.step_number === route.steps.length;
            const borderColor = isFinalProduct ? 'border-2 border-blue-500' : '';
            const textColor = isFinalProduct ? 'text-blue-400' : '';
            const titleText = isFinalProduct ? 'Target Molecule' : `Intermediate`;
            vizHtml += `
                <div class="reaction-arrow flex-shrink-0 mx-4 my-4 md:my-0"><i class="fas fa-long-arrow-alt-right text-2xl text-gray-500"></i></div>
                <div class="molecule-display ${borderColor} flex-shrink-0">
                    <div class="text-center p-4">
                        <div class="text-xs text-gray-400 mb-1">Step ${step.step_number}: ${step.yield.toFixed(1)}% yield</div>
                        <div class="font-bold ${textColor}">${titleText}</div>
                        <div class="text-xs text-gray-400 mt-1">${step.product.formula}</div>
                    </div>
                </div>`;
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
                </div>`;
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
            </div>`;
    }

    function renderLiteratureResults(papers) {
        if (!papers || papers.length === 0) {
            literatureResultsContainer.innerHTML = '<p class="text-center text-gray-400">No relevant papers found for your query in chemistry-related categories.</p>';
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
                            <span class="text-xs bg-blue-900 text-blue-300 px-2 py-1 rounded">Category: ${paper.impact || 'N/A'}</span>
                        </div>
                    </div>
                    <div class="text-sm text-gray-300 mb-3">${renderedAbstract}</div>
                    <div class="flex flex-wrap gap-2">${keywordsHtml}</div>
                </div>`;
        });
        literatureResultsContainer.innerHTML = html;
    }

    // <<< NEW RENDER FUNCTIONS FOR SOURCING AND COST >>>
    function renderSourcingInfo(routeId) {
        const data = sourcingData[routeId];
        if (!data) {
            sourcingContent.innerHTML = '<p class="text-center text-gray-400"><i class="fas fa-spinner fa-spin mr-2"></i>Loading sourcing data...</p>';
            return;
        }
        if (data.error) {
            sourcingContent.innerHTML = `<p class="text-center text-red-400">${data.error}</p>`;
            return;
        }

        let tableRows = '';
        const details = data.sourcing_details || {};
        for (const smiles in details) {
            const reagent = details[smiles];
            const cheapest = reagent.cheapest_option;
            const supplierInfo = cheapest 
                ? `${cheapest.vendor} (${formatCurrency(cheapest.price_per_g)}/g)`
                : '<span class="text-yellow-400">Not Found</span>';

            tableRows += `
                <tr class="border-b border-gray-700">
                    <td class="p-4">${reagent.formula}</td>
                    <td class="p-4 font-mono text-xs">${smiles}</td>
                    <td class="p-4">${(reagent.required_amount_g || 0).toFixed(3)} g</td>
                    <td class="p-4">${supplierInfo}</td>
                    <td class="p-4">
                        <span class="text-sm bg-gray-600 text-gray-300 px-2 py-1 rounded">${reagent.suppliers.length} available</span>
                    </td>
                </tr>`;
        }

        sourcingContent.innerHTML = `
            <h2 class="text-xl font-bold mb-4">Material Sourcing for Starting Materials</h2>
            <p class="text-sm text-gray-400 mb-6">${data.assumptions || ''}</p>
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-xs text-gray-300 uppercase">
                        <tr>
                            <th class="p-4">Reagent</th>
                            <th class="p-4">SMILES</th>
                            <th class="p-4">Amount Required</th>
                            <th class="p-4">Cheapest Supplier</th>
                            <th class="p-4">Total Suppliers</th>
                        </tr>
                    </thead>
                    <tbody class="text-sm">${tableRows}</tbody>
                </table>
            </div>`;
    }

    function renderCostAnalysis(routeId) {
        const data = sourcingData[routeId];
        if (!data) {
            costContent.innerHTML = '<p class="text-center text-gray-400"><i class="fas fa-spinner fa-spin mr-2"></i>Loading cost data...</p>';
            return;
        }
        if (data.error) {
            costContent.innerHTML = `<p class="text-center text-red-400">${data.error}</p>`;
            return;
        }

        let tableRows = '';
        const details = data.sourcing_details || {};
        for (const smiles in details) {
            const reagent = details[smiles];
            tableRows += `
                <tr class="border-b border-gray-700">
                    <td class="p-4">${reagent.formula}</td>
                    <td class="p-4">${(reagent.required_amount_g || 0).toFixed(3)} g</td>
                    <td class="p-4">${formatCurrency(reagent.cheapest_option?.price_per_g || 0)}</td>
                    <td class="p-4 font-bold">${formatCurrency(reagent.estimated_cost || 0)}</td>
                </tr>`;
        }

        costContent.innerHTML = `
            <h2 class="text-xl font-bold mb-6">Estimated Cost Analysis</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div class="bg-gray-700 rounded-lg p-6 text-center">
                    <h3 class="text-sm font-medium text-gray-400 uppercase">Total Estimated Cost</h3>
                    <p class="text-4xl font-bold mt-2 gradient-text-blue">${formatCurrency(data.total_cost || 0)}</p>
                </div>
                 <div class="md:col-span-2 bg-gray-700 rounded-lg p-6">
                    <h3 class="text-sm font-medium text-gray-400 uppercase mb-3">Assumptions</h3>
                    <p class="text-sm text-gray-300">${data.assumptions || ''}</p>
                </div>
            </div>
            <h3 class="text-lg font-bold mb-4">Cost Breakdown by Reagent</h3>
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                     <thead class="bg-gray-700 text-xs text-gray-300 uppercase">
                        <tr>
                            <th class="p-4">Reagent</th>
                            <th class="p-4">Amount Required</th>
                            <th class="p-4">Cost per Gram</th>
                            <th class="p-4">Subtotal</th>
                        </tr>
                    </thead>
                    <tbody class="text-sm">${tableRows}</tbody>
                </table>
            </div>`;
    }
    
    function formatCurrency(value) {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
    }
});