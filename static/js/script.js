// Global variables
let synthesisRoutesData = [];
let literatureResults = [];
let sourcingData = {};
let currentTargetSMILES = '';
let cy; // Cytoscape instance

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
    const knowledgeContent = document.getElementById('knowledge-content');

    const progressBar = document.getElementById('project-progress-bar');
    const progressStepsContainer = document.getElementById('project-progress-steps');
    const mainTabs = document.getElementById('main-tabs');

    const newRouteModal = document.getElementById('new-route-modal');
    const cancelNewRouteBtn = document.getElementById('cancel-new-route-btn');
    const submitNewRouteBtn = document.getElementById('submit-new-route-btn');
    const newRouteSuggestionInput = document.getElementById('new-route-suggestion-input');

    // --- Selectors for Save/Open Project --- 
    const saveProjectBtn = document.getElementById('save-project-btn');
    const openProjectBtn = document.getElementById('open-project-btn');
    const projectFileInput = document.getElementById('project-file-input');

    // --- Initial State Setup ---
    updateProgress(0);
    setupTabControls();

    // --- Event Listeners ---
    searchButton.addEventListener('click', handleSearch);
    literatureSortSelect.addEventListener('change', handleSortLiterature);
    newRouteButton.addEventListener('click', handleNewRouteClick);
    cancelNewRouteBtn.addEventListener('click', () => newRouteModal.classList.add('hidden'));
    submitNewRouteBtn.addEventListener('click', handleNewRouteSubmit);

    // --- Event listeners for Save/Open
    saveProjectBtn.addEventListener('click', handleSaveProject);
    openProjectBtn.addEventListener('click', handleOpenProject);
    projectFileInput.addEventListener('change', loadProjectFromFile);

    // --- Dynamic event listeners for expanding reactants/visualizations ---
    routeContentContainer.addEventListener('click', function(e) {
        if (e.target.matches('.expand-reactant-btn')) {
            e.preventDefault();
            const button = e.target;
            const smiles = button.dataset.smiles;
            const targetContainerId = button.dataset.target;
            handleExpandReactant(smiles, targetContainerId, button);
        }
    });

    routeContentContainer.addEventListener('click', function(e) {
        const expandableBox = e.target.closest('.expandable-molecule');
        if (expandableBox) {
            e.preventDefault();
            const smiles = expandableBox.dataset.smiles;
            const targetContainerId = expandableBox.dataset.targetId;
            handleExpandVisualization(smiles, targetContainerId, expandableBox);
        }
    });

    // ---  Project Save/Load ---
    function handleSaveProject() {
        if (!currentTargetSMILES) {
            alert("There is no active project to save. Please search for a molecule first.");
            return;
        }

        // 1. Gather all state data into a single object
        const projectData = {
            savedAt: new Date().toISOString(),
            // Core data
            literatureResults: literatureResults,
            synthesisRoutesData: synthesisRoutesData,
            sourcingData: sourcingData,
            currentTargetSMILES: currentTargetSMILES,
            // UI state for better restoration
            targetIdentifier: targetMoleculeInput.value,
            searchKeywords: searchTermsInput.value,
            targetMoleculeName: document.getElementById('project-target-molecule-name').textContent,
        };

        // 2. Convert to a JSON string
        const jsonString = JSON.stringify(projectData, null, 2); // Pretty-print the JSON
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        // 3. Create a temporary link to trigger the download
        const a = document.createElement('a');
        a.href = url;
        const safeName = (projectData.targetIdentifier || 'untitled').replace(/[^a-z0-9]/gi, '_').toLowerCase();
        a.download = `chemsynth_project_${safeName}.json`;
        document.body.appendChild(a);
        a.click();

        // 4. Clean up
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Visual feedback
        const originalText = saveProjectBtn.innerHTML;
        saveProjectBtn.innerHTML = '<i class="fas fa-check mr-2"></i> Saved!';
        setTimeout(() => {
            saveProjectBtn.innerHTML = originalText;
        }, 2000);
    }

    /**
     * Triggers the hidden file input dialog to allow the user to select a project file.
     */
    function handleOpenProject() {
        projectFileInput.click();
    }

    /**
     * Reads the selected JSON file and initiates the UI restoration process.
     * @param {Event} event - The file input change event.
     */
    function loadProjectFromFile(event) {
        const file = event.target.files[0];
        if (!file) {
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const projectData = JSON.parse(e.target.result);
                loadProjectDataIntoUI(projectData);
            } catch (error) {
                console.error("Error parsing project file:", error);
                alert("Failed to load project. The file may be corrupted or not a valid ChemSynthAI project file.");
            }
        };
        reader.readAsText(file);
        
        // Reset the input so the 'change' event fires again if the same file is selected
        event.target.value = null; 
    }

    /**
     * Restores the entire application state and UI from a project data object.
     * @param {object} projectData - The parsed data from a saved project JSON file.
     */
    function loadProjectDataIntoUI(projectData) {
        console.log("Loading project data...", projectData);
        resetUI();

        // 1. Restore global state variables
        literatureResults = projectData.literatureResults || [];
        synthesisRoutesData = projectData.synthesisRoutesData || [];
        sourcingData = projectData.sourcingData || {};
        currentTargetSMILES = projectData.currentTargetSMILES || '';

        // 2. Restore UI elements
        targetMoleculeInput.value = projectData.targetIdentifier || '';
        searchTermsInput.value = projectData.searchKeywords || '';
        document.getElementById('project-target-molecule-name').textContent = projectData.targetMoleculeName || 'Project Loaded';
        document.getElementById('project-target-molecule-formula').textContent = projectData.currentTargetSMILES || '';
        
        // 3. Re-render all sections with the loaded data
        renderLiteratureResults(literatureResults);
        
        if (synthesisRoutesData.length > 0) {
            const firstRouteId = synthesisRoutesData[0].id;
            synthesisContent.classList.remove('hidden');
            renderRouteTabs(synthesisRoutesData, firstRouteId);
            renderSingleRoute(synthesisRoutesData[0]);

            // Since we have all data, we can directly render other tabs' content.
            // The tab switching logic will handle showing/hiding them.
            renderSourcingInfo(firstRouteId);
            renderCostAnalysis(firstRouteId);
            renderKnowledgeGraph(firstRouteId);
        }
        
        // 4. Update progress and switch to a relevant tab
        updateProgress(5); // Mark all stages as complete
        document.querySelector('.tab-link[data-tab="synthesis"]').click();
    }

    // --- Main Handler Functions ---
    async function handleSearch(event) {
        event.preventDefault();
        const identifier = targetMoleculeInput.value.trim();
        const keywords = searchTermsInput.value.trim();

        if (!identifier) {
            alert("Please enter a target molecule identifier (SMILES, Name, or CAS).");
            return;
        }

        resetUI();
        updateProgress(0, true);

        try {
            const resolvedData = await resolveIdentifier(identifier);
            if (resolvedData) {
                const literatureQuery = (resolvedData.name + ' ' + keywords).trim();
                currentTargetSMILES = resolvedData.smiles;
                document.getElementById('project-target-molecule-name').textContent = `Target: ${resolvedData.name}`;
                document.getElementById('project-target-molecule-formula').textContent = resolvedData.smiles;
                fetchLiterature(literatureQuery);
                fetchSynthesisPlan(identifier);
            }
        } catch (error) {
            alert(error.message);
            updateProgress(0);
        }
    }

    function resetUI() {
        synthesisRoutesData = [];
        literatureResults = [];
        sourcingData = {};
        currentTargetSMILES = '';
        if (cy) {
            cy.destroy();
            cy = null;
        }
        document.getElementById('cy-knowledge-graph').innerHTML = '<p class="text-center text-gray-500 pt-16">Select a synthesis route to generate its knowledge graph.</p>';
        routeContentContainer.innerHTML = '';
        routeTabsContainer.innerHTML = '';
        sourcingContent.innerHTML = '<p class="text-center text-gray-500 py-16">Perform a synthesis plan search to see material sourcing information.</p>';
        costContent.innerHTML = '<p class="text-center text-gray-500 py-16">Perform a synthesis plan search to see cost analysis.</p>';
    }

    function handleNewRouteClick() {
        if (!currentTargetSMILES) {
            alert("Please perform a search for a target molecule first.");
            return;
        }
        newRouteModal.classList.remove('hidden');
        newRouteSuggestionInput.focus();
    }

    async function handleNewRouteSubmit() {
        const suggestion = newRouteSuggestionInput.value.trim();
        if (!suggestion) {
            alert("Please provide a suggestion to guide the AI.");
            return;
        }

        newRouteModal.classList.add('hidden');
        newRouteSuggestionInput.value = '';
        generateNewRouteApiCall(suggestion);
    }

    async function generateNewRouteApiCall(suggestion) {
        newRouteButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Generating...';
        newRouteButton.disabled = true;

        try {
            const response = await fetch('/api/generate_new_route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    suggestion: suggestion,
                    target_smiles: currentTargetSMILES,
                    existing_routes: synthesisRoutesData // Pass existing routes for context
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to generate new route.');

            if (data.new_route) {
                addNewRouteToDisplay(data.new_route);
            } else {
                 throw new Error(data.error || "API did not return a valid new route object.");
            }

        } catch (error) {
            console.error('Error generating new route:', error);
            alert(`Error: ${error.message}`);
        } finally {
            newRouteButton.innerHTML = '<i class="fas fa-plus mr-2"></i> New Route';
            newRouteButton.disabled = false;
        }
    }

    function handleSortLiterature() {
        const sortBy = literatureSortSelect.value;
        let sortedResults = [...literatureResults];

        if (sortBy === 'date') {
            sortedResults.sort((a, b) => new Date(b.published) - new Date(a.published));
        } else if (sortBy === 'citations') {
            // Assuming relevance_score is a proxy for citations/importance
            sortedResults.sort((a, b) => (b.relevance_score || 0) - (a.relevance_score || 0));
        } else {
             // Default to original relevance sort
             sortedResults = [...literatureResults];
        }
        renderLiteratureResults(sortedResults);
    }

    function addNewRouteToDisplay(newRoute) {
        // Ensure a unique ID for the new route
        newRoute.id = newRoute.id || `route_chemfm_${Date.now()}`;
        synthesisRoutesData.push(newRoute);
        fetchSourcingAndCost([newRoute]); // Fetch sourcing just for the new route
        renderRouteTabs(synthesisRoutesData, newRoute.id);
        renderSingleRoute(newRoute);
        // If knowledge tab is active, render the new graph
        if (document.querySelector('#main-tabs .tab-link[data-tab="knowledge"].border-blue-500')) {
            renderKnowledgeGraph(newRoute.id);
        }
    }

    async function handleExpandReactant(smiles, targetContainerId, button) {
        const container = document.getElementById(targetContainerId);
        if (!container) return;

        // Prevent multiple clicks
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i> Loading...';

        try {
            const response = await fetch('/api/plan_synthesis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier: smiles }),
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to fetch sub-route.");

            if (data.routes && data.routes.length > 0) {
                // Render the best sub-route
                renderSubRoute(data.routes[0], container);
                button.style.display = 'none'; // Hide button after successful expansion
            } else {
                container.innerHTML = `<p class="text-xs text-green-400 italic mt-2">This is a stock material (no further synthesis found).</p>`;
                button.style.display = 'none';
            }

        } catch (error) {
            console.error('Error expanding synthesis tree:', error);
            container.innerHTML = `<p class="text-xs text-red-400 italic mt-2">Error: ${error.message}</p>`;
            button.disabled = false; // Re-enable button on failure
            button.innerHTML = '<i class="fas fa-search-plus mr-1"></i> Expand';
        }
    }

    async function handleExpandVisualization(smiles, targetContainerId, clickedBox) {
        const container = document.getElementById(targetContainerId);
        if (!container || clickedBox.classList.contains('is-expanded')) return;

        clickedBox.classList.add('is-loading'); // Visual feedback
        clickedBox.classList.remove('expandable-molecule'); // Prevent re-clicks

        try {
            const response = await fetch('/api/plan_synthesis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier: smiles }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to fetch sub-route.");

            if (data.routes && data.routes.length > 0) {
                renderSubRouteVisualization(data.routes[0], container);
            } else {
                container.innerHTML = `<p class="text-xs text-green-400 italic mt-2 text-center">Stock Material</p>`;
            }
            clickedBox.classList.add('is-expanded');
        } catch (error) {
            console.error('Error expanding synthesis viz:', error);
            container.innerHTML = `<p class="text-xs text-red-400 italic mt-2 text-center">Error: ${error.message}</p>`;
            clickedBox.classList.add('expandable-molecule'); // Allow retry on error
        } finally {
            clickedBox.classList.remove('is-loading');
        }
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
        // --- MODIFICATION: Updated initial loading message ---
        routeContentContainer.innerHTML = '<p class="text-center text-gray-400 py-8"><i class="fas fa-atom fa-spin mr-2"></i>Performing retrosynthesis analysis...</p>';

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
                const firstRoute = synthesisRoutesData[0];
                
                // Render tabs first
                renderRouteTabs(synthesisRoutesData, firstRoute.id);
                
                // Show optimization animation
                const optimizationHtml = `
                    ${renderRouteVisualization(firstRoute)}
                    <div class="text-center py-16 animate-pulse">
                        <i class="fas fa-cogs text-blue-400 text-3xl"></i>
                        <p class="mt-4 text-lg text-gray-300">Optimizing yields and reaction conditions...</p>
                        <p class="text-sm text-gray-500">Running advanced yield prediction and LLM-based procedural generation.</p>
                    </div>
                `;
                
                routeContentContainer.innerHTML = optimizationHtml;
                updateProgress(2, true);
                
                // After delay, show final results
                setTimeout(() => {
                    renderSingleRoute(firstRoute);
                    updateProgress(2);
                    fetchSourcingAndCost(synthesisRoutesData);
                }, 2500); // wait for 2.5 seconds
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
        }
    }
    
    async function fetchSourcingAndCost(routes) {
        updateProgress(3, true);
        const analysisPromises = routes.map(route =>
            fetch('/api/analyze_sourcing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // body: JSON.stringify({ route_steps: route.steps, target_amount_g: 1.0 })
                body: JSON.stringify({ 
                    route_steps: route.steps, 
                    target_amount_g: 1.0,      
                    default_yield_percent: 85.0 
                })
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

        updateProgress(4);
        
        // After sourcing is done, refresh the active tab if it's sourcing/cost/knowledge
        const activeMainTab = document.querySelector('#main-tabs .tab-link.border-blue-500');
        const activeRouteTab = document.querySelector('.route-tab.border-blue-500');
        if (activeMainTab && activeRouteTab) {
            const tabName = activeMainTab.dataset.tab;
            const routeId = activeRouteTab.dataset.routeId;
            if (tabName === 'sourcing') renderSourcingInfo(routeId);
            if (tabName === 'cost') renderCostAnalysis(routeId);
            if (tabName === 'knowledge') renderKnowledgeGraph(routeId);
        }
    }

    // --- UI Control and Rendering ---
    function setupTabControls() {
        mainTabs.addEventListener('click', (e) => {
            e.preventDefault();
            const clickedTab = e.target.closest('.tab-link');
            if (!clickedTab) return;

            const tabName = clickedTab.dataset.tab;

            document.querySelectorAll('#main-tabs .tab-link').forEach(tab => {
                tab.classList.remove('border-blue-500', 'text-blue-500');
                tab.classList.add('border-transparent', 'text-gray-500');
            });
            clickedTab.classList.add('border-blue-500', 'text-blue-500');
            clickedTab.classList.remove('border-transparent', 'text-gray-500');

            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.add('hidden');
            });
            const activeContent = document.getElementById(`${tabName}-content`);
            if (activeContent) activeContent.classList.remove('hidden');

            if (['synthesis', 'sourcing', 'cost', 'knowledge'].includes(tabName)) {
                const activeRouteTab = document.querySelector('.route-tab.border-blue-500');
                if (activeRouteTab) {
                    const activeRouteId = activeRouteTab.dataset.routeId;
                    if (tabName === 'sourcing') renderSourcingInfo(activeRouteId);
                    if (tabName === 'cost') renderCostAnalysis(activeRouteId);
                    if (tabName === 'knowledge') renderKnowledgeGraph(activeRouteId);
                }
            }
        });
    }

    function updateProgress(stage, isLoading = false) {
        const stages = ['Literature', 'Synthesis', 'Sourcing', 'Costing', 'Knowledge'];
        let width = stage * 25; // 0, 25, 50, 75, 100
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
        if(stage >= 4) updateProgress(5); // Complete knowledge stage
    }

    function renderRouteTabs(routes, activeRouteId) {
        let tabsHtml = `<nav class="-mb-px flex space-x-8 overflow-x-auto">`;
        routes.forEach((route, index) => {
            const isHypothetical = route.id.includes('chemfm') || route.id.includes('hypothetical');
            const defaultName = isHypothetical ? `ChemFM Route ${index - (routes.length - 1) + 1}` : `Route ${String.fromCharCode(65 + index)}`;
            const routeName = route.name || defaultName;
            const isActive = route.id === activeRouteId ? 'border-blue-500 text-blue-500' : 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300';

            const rawYield = route.overall_yield || 0;
            // The yield is already a percentage from the backend
            const displayYield = rawYield;

            tabsHtml += `<a href="#" class="${isActive} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm route-tab" data-route-id="${route.id}">
                            ${routeName} (${displayYield.toFixed(1)}% yield)
                        </a>`;
        });
        tabsHtml += '</nav>';
        routeTabsContainer.innerHTML = tabsHtml;

        document.querySelectorAll('.route-tab').forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                const routeId = this.dataset.routeId;
                const routeData = synthesisRoutesData.find(r => r.id === routeId);
                if (!routeData) return;

                renderRouteTabs(synthesisRoutesData, routeId);
                renderSingleRoute(routeData);

                const activeMainTab = document.querySelector('#main-tabs .tab-link.border-blue-500');
                if (activeMainTab) {
                    const tabName = activeMainTab.dataset.tab;
                    if (tabName === 'sourcing') renderSourcingInfo(routeId);
                    if (tabName === 'cost') renderCostAnalysis(routeId);
                    if (tabName === 'knowledge') renderKnowledgeGraph(routeId);
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

    // function renderRouteVisualization(route) {
    //     if (!route.steps || route.steps.length === 0) return '';
    //     let vizHtml = '<div class="reaction-visualization mb-6"><div class="flex items-center justify-start mb-8 overflow-x-auto p-4">';
        
    //     const firstReactants = route.steps[0].reactants;
    //     const reactantNames = firstReactants.map(r => r.formula || 'Reactant').join(' + ');
    //     vizHtml += `
    //         <div class="molecule-display flex-shrink-0">
    //             <div class="text-center p-4">
    //                 <div class="text-xs text-gray-400 mb-1">Starting Material(s)</div>
    //                 <div class="font-bold">${reactantNames}</div>
    //             </div>
    //         </div>`;

    //     route.steps.forEach(step => {
    //         const isFinalProduct = step.product.smiles === currentTargetSMILES;
    //         const borderColor = isFinalProduct ? 'border-2 border-blue-500' : '';
    //         const textColor = isFinalProduct ? 'text-blue-400' : '';
    //         const titleText = isFinalProduct ? 'Target Molecule' : `Intermediate`;
    //         vizHtml += `
    //             <div class="reaction-arrow flex-shrink-0 mx-4 my-4 md:my-0"><i class="fas fa-long-arrow-alt-right text-2xl text-gray-500"></i></div>
    //             <div class="molecule-display ${borderColor} flex-shrink-0">
    //                 <div class="text-center p-4">
    //                     <div class="text-xs text-gray-400 mb-1">Step ${step.step_number}: ${(step.yield || 0).toFixed(1)}% yield</div>
    //                     <div class="font-bold ${textColor}">${titleText}</div>
    //                     <div class="text-xs text-gray-400 mt-1">${step.product.formula}</div>
    //                 </div>
    //             </div>`;
    //     });
    //     vizHtml += '</div></div>';
    //     return vizHtml;
    // }

    function renderRouteVisualization(route) {
        if (!route.steps || route.steps.length === 0) return '';
        
        let vizHtml = '<div class="reaction-visualization mb-6"><div class="flex items-start justify-start mb-8 overflow-x-auto p-4">';
        
        // This is the main wrapper for each step's visualization block
        const renderStepBlock = (step, isInitial = false) => {
            const product = step.product;
            const reactants = step.reactants;
            const isFinalProduct = product.smiles === currentTargetSMILES;
            const containerId = `sub-viz-${route.id}-${step.step_number}`;

            let blockHtml = '';

            // Render the starting material(s) box only for the very first step
            if (isInitial) {
                const reactantNames = reactants.map(r => r.formula || 'Reactant').join(' + ');
                const initialReactantSmiles = reactants.map(r => r.smiles).join('.'); // Join for potential multi-reactant expansion
                
                blockHtml += `
                <div class="flex flex-col items-center">
                    <div class="molecule-display flex-shrink-0 expandable-molecule" 
                         data-smiles="${initialReactantSmiles}" 
                         data-target-id="sub-viz-${route.id}-initial">
                        <div class="text-center p-4">
                            <div class="text-xs text-gray-400 mb-1">Starting Material(s)</div>
                            <div class="font-bold">${reactantNames}</div>
                        </div>
                    </div>
                    <div id="sub-viz-${route.id}-initial" class="w-full mt-2"></div>
                </div>`;
            }

            // Render the arrow and the product box for the current step
            const expandableClass = isFinalProduct ? '' : 'expandable-molecule';
            const borderColor = isFinalProduct ? 'border-2 border-blue-500' : '';
            const textColor = isFinalProduct ? 'text-blue-400' : '';
            const titleText = isFinalProduct ? 'Target Molecule' : `Intermediate`;

            blockHtml += `
                <div class="reaction-arrow flex-shrink-0 mx-4 self-center"><i class="fas fa-long-arrow-alt-right text-2xl text-gray-500"></i></div>
                <div class="flex flex-col items-center">
                    <div class="molecule-display ${borderColor} ${expandableClass} flex-shrink-0" 
                         data-smiles="${product.smiles}" 
                         data-target-id="${containerId}">
                        <div class="text-center p-4">
                            <div class="text-xs text-gray-400 mb-1">Step ${step.step_number}: ${(step.yield || 0).toFixed(1)}% yield</div>
                            <div class="font-bold ${textColor}">${titleText}</div>
                            <div class="text-xs text-gray-400 mt-1">${product.formula}</div>
                        </div>
                    </div>
                    <div id="${containerId}" class="w-full mt-2"></div>
                </div>`;
            
            return blockHtml;
        };

        // Build the complete visualization
        vizHtml += renderStepBlock(route.steps[0], true); // Render the first step which includes the initial reactant
        for (let i = 1; i < route.steps.length; i++) {
            vizHtml += renderStepBlock(route.steps[i]);
        }
        
        vizHtml += '</div></div>';
        return vizHtml;
    }

    function renderSubRouteVisualization(subRoute, containerElement) {
        if (!subRoute || !subRoute.steps || subRoute.steps.length === 0) {
            containerElement.innerHTML = `<p class="text-xs text-gray-400 italic mt-2">Could not render sub-route.</p>`;
            return;
        }

        let subVizHtml = '<div class="flex items-start justify-center p-3 bg-gray-900 bg-opacity-50 rounded-md border border-gray-700">';

        // Starting material for the sub-route
        const initialReactants = subRoute.steps[0].reactants;
        const reactantNames = initialReactants.map(r => r.formula).join(' + ');
        const initialReactantSmiles = initialReactants.map(r => r.smiles).join('.');
        
        subVizHtml += `
            <div class="flex flex-col items-center">
                <div class="molecule-display-sm expandable-molecule" 
                     data-smiles="${initialReactantSmiles}" 
                     data-target-id="sub-viz-${subRoute.id}-initial">
                     <div class="text-center p-2">
                        <div class="font-bold text-xs">${reactantNames}</div>
                     </div>
                </div>
                <div id="sub-viz-${subRoute.id}-initial" class="w-full mt-2"></div>
            </div>`;

        // Each step in the sub-route
        subRoute.steps.forEach(step => {
            const product = step.product;
            const containerId = `sub-viz-${subRoute.id}-${step.step_number}`;
            
            subVizHtml += `
            <div class="reaction-arrow flex-shrink-0 mx-2 self-center"><i class="fas fa-long-arrow-alt-right text-lg text-gray-600"></i></div>
            <div class="flex flex-col items-center">
                <div class="molecule-display-sm expandable-molecule" 
                     data-smiles="${product.smiles}"
                     data-target-id="${containerId}">
                    <div class="text-center p-2">
                        <div class="text-xs text-gray-400">Yield: ${(step.yield || 0).toFixed(0)}%</div>
                        <div class="font-bold text-xs">${product.formula}</div>
                    </div>
                </div>
                <div id="${containerId}" class="w-full mt-2"></div>
            </div>`;
        });

        subVizHtml += '</div>';
        containerElement.innerHTML = subVizHtml;
    }

    function renderRouteDetails(steps) {
        if (!steps) return '';
        let detailsHtml = '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">';
        
        steps.forEach((step, stepIndex) => {
            const imageHtml = step.reaction_image_url
                ? `<div class="bg-white rounded-md p-2 mb-4"><img src="${step.reaction_image_url}" alt="Reaction diagram for step ${step.step_number}" class="w-full h-auto"/></div>`
                : `<div class="bg-gray-800 rounded-md p-2 mb-4 text-center text-xs text-gray-500">Image not available</div>`;

            // <<< NEW: Generate list of reactants with expand buttons >>>
            const reactantsHtml = step.reactants.map((reactant, reactantIndex) => {
                const subRouteContainerId = `sub-route-container-s${stepIndex}-r${reactantIndex}`;
                // Only show expand button if it's not the final target molecule
                const expandButton = reactant.smiles !== currentTargetSMILES
                    ? `<button class="expand-reactant-btn text-blue-400 hover:text-blue-300 text-xs ml-2" 
                                data-smiles="${reactant.smiles}" 
                                data-target="${subRouteContainerId}">
                           <i class="fas fa-search-plus mr-1"></i> Expand
                       </button>`
                    : '';

                return `<li class="flex justify-between items-center py-1">
                            <span class="font-mono text-xs">${reactant.formula}</span>
                            ${expandButton}
                        </li>
                        <div id="${subRouteContainerId}" class="pl-4 border-l-2 border-gray-600 ml-2"></div>`;
            }).join('');

            detailsHtml += `
                <div class="bg-gray-700 rounded-lg p-4 flex flex-col">
                    <h4 class="font-bold mb-2">Step ${step.step_number}: ${step.title}</h4>
                    <p class="text-sm text-gray-300 mb-3"><span class="font-medium">Conditions:</span> ${step.reagents_conditions}</p>
                    
                    ${imageHtml}
                    
                    <div class="flex-grow">
                        <h5 class="text-sm font-semibold mb-2">Reactants:</h5>
                        <ul class="text-sm text-gray-300 mb-3 space-y-1">
                           ${reactantsHtml}
                        </ul>
                        <div class="text-xs text-gray-400 mb-2"><span class="font-medium">Predicted Yield:</span> ${(step.yield || 0).toFixed(1)}%</div>
                        <div class="text-xs text-gray-400"><span class="font-medium">Notes:</span> ${step.source_notes}</div>
                    </div>
                </div>`;
        });
        detailsHtml += '</div>';
        return detailsHtml;
    }

    function renderSubRoute(subRoute, containerElement) {
        if (!subRoute || !subRoute.steps || subRoute.steps.length === 0) {
            containerElement.innerHTML = `<p class="text-xs text-gray-400 italic mt-2">Could not render sub-route.</p>`;
            return;
        }

        let subRouteHtml = `<div class="mt-2 p-3 bg-gray-800 rounded-md">
                               <h6 class="text-xs font-bold text-blue-300 mb-2">Sub-Synthesis (Overall Yield: ${subRoute.overall_yield.toFixed(1)}%)</h6>`;
        
        subRoute.steps.forEach((step, stepIndex) => {
            const reactantsList = step.reactants.map(r => `<span class="font-mono">${r.formula}</span>`).join(' + ');
            subRouteHtml += `
                <div class="mb-2 last:mb-0">
                    <p class="text-xs">
                        <span class="text-gray-400">Sub-Step ${step.step_number}:</span>
                        ${reactantsList} → ${step.product.formula}
                        <span class="text-gray-500">(${(step.yield || 0).toFixed(1)}%)</span>
                    </p>
                </div>
            `;
        });
        
        subRouteHtml += `</div>`;
        containerElement.innerHTML = subRouteHtml;
    }
    
    function renderRouteEvaluation(evaluation) {
        if (!evaluation || (Object.keys(evaluation).length === 0)) return '';
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
            const renderedAbstract = marked.parse(paper.abstract || 'No abstract available.');
            html += `
                <div class="bg-gray-700 rounded-lg p-4 mb-4 transition-shadow duration-300 hover:shadow-lg">
                    <div class="flex justify-between items-start">
                        <a href="${paper.url}" target="_blank" class="flex-grow">
                            <h4 class="font-bold text-blue-400 hover:text-blue-300">${paper.title}</h4>
                        </a>
                        <div class="flex items-center space-x-2 flex-shrink-0 ml-4">
                             <span class="text-xs bg-gray-600 text-gray-300 px-2 py-1 rounded-full">Score: ${(paper.relevance_score || 0).toFixed(2)}</span>
                        </div>
                    </div>
                     <p class="text-sm text-gray-400 mb-2">${paper.source}</p>
                    <div class="text-sm text-gray-300 mb-3">${renderedAbstract}</div>
                </div>`;
        });
        literatureResultsContainer.innerHTML = html;
    }

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
        if (Object.keys(details).length === 0) {
            sourcingContent.innerHTML = '<h2 class="text-xl font-bold mb-4">Material Sourcing</h2><p class="text-center text-gray-400 py-8">This route uses intermediates from previous steps. No new starting materials require sourcing.</p>';
            return;
        }

        for (const smiles in details) {
            const reagent = details[smiles];
            const cheapest = reagent.cheapest_option;
            const supplierInfo = cheapest
                ? `${cheapest.vendor} (${formatCurrency(cheapest.price_per_g)}/g)`
                : '<span class="text-yellow-400">Not Found</span>';

            tableRows += `
                <tr class="border-b border-gray-700">
                    <td class="p-4">${reagent.name || reagent.formula}</td>
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
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                    <thead class="bg-gray-700 text-xs text-gray-300 uppercase">
                        <tr><th class="p-4">Reagent</th><th class="p-4">SMILES</th><th class="p-4">Amount Required</th><th class="p-4">Cheapest Supplier</th><th class="p-4">Total Suppliers</th></tr>
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

        const route = synthesisRoutesData.find(r => r.id === routeId);
        if (!route) {
             costContent.innerHTML = `<p class="text-center text-red-400">Error: Could not find route data for ID ${routeId}.</p>`;
             return;
        }

        // --- NEW: Generate UI for per-step yield overrides ---
        let stepYieldsHtml = '';
        if (route.steps && route.steps.length > 0) {
            stepYieldsHtml = route.steps.map(step => `
                <div class="flex items-center justify-between py-2 border-b border-gray-800 last:border-b-0">
                    <label for="yield-step-${step.step_number}" class="text-sm text-gray-300">
                        Step ${step.step_number}: ${step.title}
                    </label>
                    <input type="number" step="0.1" id="yield-step-${step.step_number}" 
                           class="molecule-input w-24 p-1 rounded-lg text-right" 
                           placeholder="${(step.yield || 0).toFixed(1)}"
                           data-step-number="${step.step_number}">
                </div>
            `).join('');
        }

        // --- MODIFICATION: Updated controls layout ---
        const controlsHtml = `
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <!-- Global Parameters -->
                <div class="bg-gray-900 p-4 rounded-lg border border-gray-700">
                    <h4 class="font-bold text-gray-200 mb-3">Global Parameters</h4>
                    <div class="space-y-3">
                        <div>
                            <label for="cost-target-amount" class="block text-sm font-medium text-gray-300 mb-1">Target Amount (g)</label>
                            <input type="number" step="0.01" id="cost-target-amount" class="molecule-input w-full p-2 rounded-lg" value="${data.analysis_summary.target_amount_g || 1.0}">
                        </div>
                        <div>
                            <label for="cost-default-yield" class="block text-sm font-medium text-gray-300 mb-1">Default Yield for Unspecified Steps (%)</label>
                            <input type="number" step="0.1" id="cost-default-yield" class="molecule-input w-full p-2 rounded-lg" value="${(data.assumptions.find(s => s.includes('Default yield')) || '85').match(/[0-9.]+/)[0]}">
                        </div>
                    </div>
                </div>
                <!-- Per-Step Overrides -->
                <div class="bg-gray-900 p-4 rounded-lg border border-gray-700">
                     <h4 class="font-bold text-gray-200 mb-3">Per-Step Yield Overrides (%)</h4>
                     <div class="space-y-1 max-h-48 overflow-y-auto pr-2">${stepYieldsHtml}</div>
                </div>
            </div>
            <div class="text-center mb-8">
                 <button id="recalculate-cost-btn" class="gradient-bg-blue text-white font-medium py-3 px-8 rounded-lg hover:shadow-lg w-full sm:w-auto">
                    <i class="fas fa-sync-alt mr-2"></i> Recalculate Cost
                </button>
            </div>
        `;

        // --- MODIFICATION: Cost breakdown table rendering (no logic change here) ---
        let tableRows = '';
        const details = data.sourcing_details || {};
        for (const smiles in details) {
            const reagent = details[smiles];
            tableRows += `
                <tr class="border-b border-gray-700">
                    <td class="p-4">${reagent.name || reagent.formula}</td>
                    <td class="p-4">${(reagent.required_amount_g || 0).toFixed(3)} g</td>
                    <td class="p-4">${formatCurrency(reagent.cheapest_option?.price_per_g || 0)}</td>
                    <td class="p-4 font-bold">${formatCurrency(reagent.estimated_cost || 0)}</td>
                </tr>`;
        }
        
        // Final container assembly
        costContent.innerHTML = `
            <h2 class="text-xl font-bold mb-4">Estimated Cost Analysis</h2>
            ${controlsHtml}
            <div id="cost-results-container">
                 <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="bg-gray-700 rounded-lg p-6 text-center">
                        <h3 class="text-sm font-medium text-gray-400 uppercase">Total Estimated Cost</h3>
                        <p class="text-4xl font-bold mt-2 gradient-text-blue">${formatCurrency(data.total_cost || 0)}</p>
                    </div>
                     <div class="md:col-span-2 bg-gray-700 rounded-lg p-6">
                        <h3 class="text-sm font-medium text-gray-400 uppercase mb-3">Analysis Assumptions</h3>
                        <ul class="text-xs text-gray-300 list-disc list-inside space-y-1">
                            ${(data.assumptions || []).map(item => `<li>${item}</li>`).join('')}
                        </ul>
                    </div>
                </div>
                <h3 class="text-lg font-bold mb-4">Cost Breakdown by Reagent</h3>
                <div class="overflow-x-auto">${Object.keys(details).length > 0 ? `
                    <table class="w-full text-left">
                         <thead class="bg-gray-700 text-xs text-gray-300 uppercase">
                            <tr><th class="p-4">Reagent</th><th class="p-4">Amount Required</th><th class="p-4">Cost per Gram</th><th class="p-4">Subtotal</th></tr>
                        </thead>
                        <tbody class="text-sm">${tableRows}</tbody>
                    </table>` : `<p class="text-center text-gray-400 py-8">No starting materials require purchasing for this route.</p>`
                }</div>
            </div>`;

        // --- MODIFICATION: Updated event listener for recalculation ---
        document.getElementById('recalculate-cost-btn').addEventListener('click', async function() {
            const button = this;
            button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Recalculating...';
            button.disabled = true;

            const targetAmount = parseFloat(document.getElementById('cost-target-amount').value);
            const defaultYield = parseFloat(document.getElementById('cost-default-yield').value);
            
            // Create a deep copy of the route steps to modify
            const updatedSteps = JSON.parse(JSON.stringify(route.steps));

            // Apply per-step overrides
            updatedSteps.forEach(step => {
                const input = document.getElementById(`yield-step-${step.step_number}`);
                const overrideValue = input.value.trim();
                if (overrideValue !== '') {
                    step.yield = parseFloat(overrideValue);
                }
            });

            try {
                const response = await fetch('/api/analyze_sourcing', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        route_steps: updatedSteps, // Send the modified steps
                        target_amount_g: targetAmount,
                        default_yield_percent: defaultYield
                    })
                });
                const newData = await response.json();
                if (!response.ok) throw new Error(newData.error || 'Failed to recalculate cost');

                sourcingData[routeId] = newData;
                renderCostAnalysis(routeId); // Re-render with new data and controls

            } catch (error) {
                console.error("Recalculation error:", error);
                alert(`Error recalculating cost: ${error.message}`);
                button.innerHTML = '<i class="fas fa-sync-alt mr-2"></i> Recalculate Cost';
                button.disabled = false;
            }
        });
    }

    function renderKnowledgeGraph(routeId) {
        const route = synthesisRoutesData.find(r => r.id === routeId);
        const sourcing = sourcingData[routeId];
        const container = document.getElementById('cy-knowledge-graph');

        if (!route || !sourcing) {
            container.innerHTML = '<p class="text-center text-gray-500 pt-16">Sourcing and route data not yet available. Please wait for analysis to complete.</p>';
            return;
        }
        if (cy) { cy.destroy(); } 
        container.innerHTML = '';

        let nodes = [];
        let edges = [];
        
        const finalProduct = route.steps[route.steps.length - 1].product;
        // Add final product node
        nodes.push({ data: { id: finalProduct.smiles, label: finalProduct.formula, type: 'product' } });

        let startingMaterialsSMILES = new Set(Object.keys(sourcing.sourcing_details || {}));
        
        // Process steps to create nodes and edges
        route.steps.forEach((step, i) => {
            const reactionNodeId = `reaction_${route.id}_${step.step_number}`;
            nodes.push({ data: { id: reactionNodeId, label: `Step ${step.step_number}`, type: 'reaction' } });
            
            // Link reactants to the reaction node
            step.reactants.forEach(reactant => {
                const isStartingMaterial = startingMaterialsSMILES.has(reactant.smiles);
                nodes.push({ data: { id: reactant.smiles, label: reactant.formula, type: isStartingMaterial ? 'start' : 'intermediate' } });
                edges.push({ data: { source: reactant.smiles, target: reactionNodeId } });
            });

            // Link reaction node to its product
            if (step.product.smiles !== finalProduct.smiles) {
                nodes.push({ data: { id: step.product.smiles, label: step.product.formula, type: 'intermediate' } });
            }
            edges.push({ data: { source: reactionNodeId, target: step.product.smiles } });
        });

        // Add supplier nodes and link them to starting materials
        for (const smiles in sourcing.sourcing_details) {
            const cheapest = sourcing.sourcing_details[smiles].cheapest_option;
            if (cheapest && cheapest.vendor) {
                nodes.push({ data: { id: cheapest.vendor, label: cheapest.vendor, type: 'vendor' } });
                edges.push({ data: { source: smiles, target: cheapest.vendor, type: 'supplier' } });
            }
        }

        nodes = Array.from(new Map(nodes.map(item => [item.data.id, item])).values());
        
        cy = cytoscape({
            container: container,
            elements: { nodes, edges },
            style: [
                { selector: 'node', style: { 'label': 'data(label)', 'color': '#CBD5E0', 'font-size': '12px', 'text-valign': 'bottom', 'text-halign': 'center', 'text-margin-y': '5px', 'background-color': '#4A5568' } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#4A5568', 'target-arrow-color': '#4A5568', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier' } },
                { selector: 'node[type="product"]', style: { 'background-color': '#3B82F6', 'shape': 'diamond', 'width': 60, 'height': 60 } },
                { selector: 'node[type="start"]', style: { 'background-color': '#10B981', 'shape': 'ellipse', 'cursor': 'pointer', 'border-color': '#A7F3D0', 'border-width': 2 } },
                { selector: 'node[type="intermediate"]', style: { 'background-color': '#6366F1', 'shape': 'ellipse' } },
                { selector: 'node[type="vendor"]', style: { 'background-color': '#F59E0B', 'shape': 'round-rectangle', 'label': 'data(label)' } },
                { selector: 'node[type="reaction"]', style: { 'background-color': '#9CA3AF', 'shape': 'rectangle', 'width': 30, 'height': 30 } },
                { selector: 'edge[type="supplier"]', style: { 'line-style': 'dashed', 'line-color': '#F59E0B' } },
                { selector: 'node.loading', style: { 'border-color': '#60A5FA', 'border-width': 4, 'border-style': 'double', 'transition-property': 'border-color, border-width', 'transition-duration': '0.5s' } }
            ],
            layout: { name: 'dagre', rankDir: 'LR', spacingFactor: 1.1 }
        });

        cy.on('tap', 'node[type="start"]', async function(evt){
            const node = evt.target;
            const smiles = node.id();

            if (node.hasClass('loading')) return; // Prevent multiple clicks

            console.log(`Expanding synthesis for starting material: ${smiles}`);
            node.addClass('loading');

            try {
                const response = await fetch('/api/plan_synthesis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ identifier: smiles }),
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Failed to fetch sub-route.");

                if (data.routes && data.routes.length > 0) {
                    const subRoute = data.routes[0]; // Take the best sub-route
                    expandGraphWithSubRoute(node, subRoute);
                } else {
                    alert(`No further synthesis route could be found for ${smiles}. This is likely a stock material.`);
                }
            } catch (error) {
                console.error('Error expanding synthesis tree:', error);
                alert(`Could not expand the synthesis tree: ${error.message}`);
            } finally {
                node.removeClass('loading');
            }
        });

        updateProgress(5); // Mark Knowledge stage as complete
    }
    
    function expandGraphWithSubRoute(clickedNode, subRoute) {
        if (!cy || !subRoute.steps || subRoute.steps.length === 0) {
            return;
        }

        // Change the clicked node from a 'start' material to an 'intermediate'
        clickedNode.removeClass('start');
        clickedNode.removeStyle('cursor border-color border-width'); // Remove clickable styling
        clickedNode.data('type', 'intermediate');

        const newElements = [];

        subRoute.steps.forEach((step, i) => {
            const subReactionId = `sub_reaction_${clickedNode.id()}_${i}`;
            
            // This reaction produces our (previously) starting material
            newElements.push({ group: 'nodes', data: { id: subReactionId, label: `Sub ${i + 1}`, type: 'reaction' } });
            // The product of this sub-reaction is the node we clicked
            newElements.push({ group: 'edges', data: { source: subReactionId, target: clickedNode.id() } });
            
            // Add the new reactants for this sub-reaction
            step.reactants.forEach(reactant => {
                // Only add the node if it doesn't already exist in the graph
                if (cy.getElementById(reactant.smiles).empty()) {
                    newElements.push({
                        group: 'nodes',
                        data: {
                            id: reactant.smiles,
                            label: reactant.formula,
                            type: 'start' // These are the NEW starting materials, making them clickable
                        }
                    });
                }
                 // Connect the new reactant to the sub-reaction
                newElements.push({ group: 'edges', data: { source: reactant.smiles, target: subReactionId } });
            });
        });

        // Add the new elements and re-run the layout
        cy.add(newElements);
        cy.layout({
            name: 'dagre',
            rankDir: 'LR',
            spacingFactor: 1.1,
            animate: true,
            animationDuration: 500
        }).run();
    }

    function formatCurrency(value) {
        if (typeof value !== 'number') return '$0.00';
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
    }
});