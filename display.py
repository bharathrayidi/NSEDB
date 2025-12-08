import pandas as pd
import numpy as np
import ipywidgets as widgets
import json
import uuid
from IPython.display import HTML, display, clear_output

# ==========================================
# 1. FOUNDRY PRO DISPLAY (Visual Stats & Context Menu)
# ==========================================
def foundry_pro_display(df: pd.DataFrame, limit_visual_rows: int = 50000):
    """
    Renders DataFrame with Foundry-like visuals.
    - Stats Panel: Bar charts & Quality metrics (Like Image 2).
    - Context Menu: Pin, Filter, Sort.
    - 100% Offline (No CDNs).
    """
    table_id = f"foundry_{uuid.uuid4().hex[:8]}"
    
    # 1. Data Prep
    df_safe = df.fillna("") 
    
    def get_friendly_type(dtype):
        s = str(dtype)
        if 'int' in s: return 'Integer'
        if 'float' in s: return 'Double'
        if 'date' in s: return 'Date'
        if 'bool' in s: return 'Boolean'
        return 'String'

    # Column Config
    cols_config = [{
        'name': str(c), 
        'type': get_friendly_type(df[c].dtype), 
        'idx': i,      
        'pinned': False 
    } for i, c in enumerate(df.columns)]
    
    # Serialize Data
    try:
        json_data = df_safe.to_json(orient='values', date_format='iso')
    except:
        json_data = json.dumps(df_safe.values.tolist(), default=str)
    
    json_cols = json.dumps(cols_config)
    total_rows = len(df)
    total_cols = len(df.columns)
    
    if total_rows > limit_visual_rows:
        showing_text = f"Showing top {limit_visual_rows:,} of {total_rows:,} rows (Visual Limit)"
    else:
        showing_text = f"Showing all {total_rows:,} rows"

    # --- 2. CSS STYLING ---
    style = f"""
    <style>
        #{table_id}_wrapper {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            border: 1px solid #d1d5db; background: #fff; font-size: 12px; margin-bottom: 20px;
            position: relative; user-select: none;
        }}
        .f-toolbar {{ border-bottom: 1px solid #e5e7eb; padding: 0 10px; height: 40px; display: flex; align-items: center; background: #fff; }}
        .f-tab {{ color: #1f6bc0; font-weight: 600; border-bottom: 2px solid #1f6bc0; height: 100%; display: flex; align-items: center; padding: 0 12px; cursor: pointer; }}
        .f-statusbar {{ background: #f7f8fa; border-bottom: 1px solid #d1d5db; padding: 6px 12px; display: flex; justify-content: flex-end; align-items: center; gap: 15px; color: #5f6b7c; font-size: 11px; }}
        
        .f-search input {{ border: 1px solid #d1d5db; padding: 4px 8px 4px 24px; font-size: 12px; width: 200px; border-radius: 3px; outline: none; }}
        .f-search {{ position: relative; }}
        .f-search:before {{ content: "🔍"; position: absolute; left: 6px; top: 4px; opacity: 0.5; }}
        
        /* Table */
        .f-scroll {{ overflow: auto; max-height: 500px; position: relative; }}
        table.f-table {{ border-collapse: separate; border-spacing: 0; width: 100%; min-width: 100%; table-layout: fixed; }}
        
        .f-table thead th {{
            position: sticky; top: 0; z-index: 20; background: #f7f8fa;
            border-bottom: 1px solid #d1d5db; border-right: 1px solid #d1d5db;
            padding: 0; height: 50px; text-align: left; vertical-align: top; cursor: pointer;
        }}
        .f-table thead th:hover {{ background: #eff6ff; }}
        .f-table thead th.is-pinned {{ background: #f0f6ff; border-bottom: 2px solid #1f6bc0; }} 
        .f-table thead th.is-filtered .th-name {{ color: #1f6bc0; }}
        
        .th-content {{ padding: 6px 10px; display: flex; flex-direction: column; justify-content: center; height: 100%; position: relative; }}
        .th-name {{ font-weight: 700; color: #10161a; margin-bottom: 2px; }}
        .th-type {{ font-size: 10px; color: #8a9ba8; font-family: monospace; text-transform: uppercase; }}
        .th-icons {{ position: absolute; top: 6px; right: 6px; font-size: 10px; color: #1f6bc0; display: flex; gap: 4px; }}
        
        .f-idx-head {{ width: 45px; min-width: 45px; z-index: 25 !important; left: 0; border-right: 2px solid #d1d5db !important; }}
        .f-idx-cell {{ position: sticky; left: 0; z-index: 10; background: #f7f8fa; border-right: 2px solid #d1d5db; text-align: right; color: #5f6b7c; font-weight: 600; }}
        
        .f-table tbody td {{
            padding: 5px 10px; border-bottom: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #333;
        }}
        .f-table tbody tr:hover td {{ background: #f0f4ff; }}
        .f-table tbody tr.is-selected td {{ background: #e5efff; }}

        /* Menu */
        .f-menu {{
            display: none; position: absolute; background: white; border: 1px solid #d1d5db;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 4px; z-index: 99999;
            min-width: 200px; padding: 0; font-family: sans-serif;
        }}
        .f-menu-item {{ padding: 8px 16px; cursor: pointer; display: flex; align-items: center; color: #10161a; transition: 0.1s; }}
        .f-menu-item:hover {{ background: #eff6ff; color: #1f6bc0; }}
        .f-menu-icon {{ width: 20px; text-align: center; margin-right: 8px; color: #5f6b7c; }}
        .f-menu-item:hover .f-menu-icon {{ color: #1f6bc0; }}
        .f-divider {{ height: 1px; background: #e5e7eb; margin: 4px 0; }}
        .f-menu-input-wrap {{ padding: 8px; border-bottom: 1px solid #eee; background: #fafafa; }}
        .f-menu-input {{ width: 100%; box-sizing: border-box; padding: 5px; border: 1px solid #ccc; border-radius: 3px; font-size: 12px; outline: none; }}
        .f-menu-input:focus {{ border-color: #1f6bc0; }}

        /* STATS MODAL (The Image 2 Look) */
        .f-modal {{
            display: none; position: absolute; top: 100px; left: 50%; transform: translateX(-50%);
            width: 700px; background: white; border: 1px solid #d1d5db;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2); z-index: 99999; border-radius: 4px; overflow: hidden;
        }}
        .f-modal-header {{ background: #f7f8fa; padding: 10px 15px; border-bottom: 1px solid #d1d5db; display: flex; justify-content: space-between; align-items: center; }}
        .f-modal-title {{ font-weight: 700; color: #10161a; font-size: 13px; }}
        .f-modal-close {{ cursor: pointer; color: #5f6b7c; font-weight: bold; }}
        
        .f-modal-body {{ display: flex; height: 350px; }}
        .stats-left {{ width: 25%; border-right: 1px solid #eee; padding: 15px; background: #fafafa; font-size: 12px; }}
        .stats-right {{ width: 75%; padding: 15px; overflow-y: auto; }}
        
        .stat-metric {{ display: flex; justify-content: space-between; margin-bottom: 8px; color: #333; }}
        .stat-metric label {{ color: #5f6b7c; }}
        .stat-metric span {{ font-weight: 600; }}
        
        /* Bar Charts */
        .bar-row {{ display: flex; align-items: center; margin-bottom: 6px; font-size: 11px; }}
        .bar-label {{ width: 130px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: right; padding-right: 10px; color: #333; }}
        .bar-track {{ flex-grow: 1; height: 18px; background: #f1f1f1; position: relative; border-radius: 2px; }}
        .bar-fill {{ height: 100%; background: #9cb4cd; border-radius: 2px; }}
        .bar-val {{ margin-left: 8px; width: 40px; color: #666; font-size: 10px; }}
    </style>
    """

    # --- 3. JAVASCRIPT LOGIC ---
    # NOTE: Javascript variables inside Python f-strings use {{ }} for objects to avoid errors.
    script = f"""
    <script>
    (function() {{
        const tableId = "{table_id}";
        const wrapper = document.getElementById(tableId + '_wrapper');
        const tbody = document.getElementById(tableId + '_tbody');
        const theadRow = document.getElementById(tableId + '_thead_row');
        const countSpan = document.getElementById(tableId + '_count');
        const menu = document.getElementById(tableId + '_menu');
        const modal = document.getElementById(tableId + '_modal');
        const searchInput = document.getElementById(tableId + '_search');
        
        let fullData = {json_data}; 
        let colMeta = {json_cols}; 
        let activeFilters = {{}}; 
        let visibleCols = colMeta; 
        let displayRows = fullData; 
        let activeColMeta = null;
        let selectedCellVal = null; 
        
        function recalcRows() {{
            displayRows = fullData.filter(row => {{
                for (let idx in activeFilters) {{
                    let filter = activeFilters[idx];
                    let cell = String(row[idx]);
                    let cellLower = cell.toLowerCase();
                    
                    if (filter.type === 'text') {{
                        if (!cellLower.includes(filter.val)) return false; 
                    }} else if (filter.type === 'include') {{
                        if (cell !== filter.val) return false;
                    }} else if (filter.type === 'exclude') {{
                        if (cell === filter.val) return false;
                    }}
                }}
                return true;
            }});
            renderBody();
            renderHeader();
        }}

        function renderHeader() {{
            let html = '<th class="f-idx-head"></th>';
            visibleCols.forEach((col, i) => {{
                let classes = [];
                if(col.pinned) classes.push('is-pinned');
                if(activeFilters[col.idx]) classes.push('is-filtered');
                let icons = '';
                if(col.pinned) icons += '<span>📌</span>';
                if(activeFilters[col.idx]) icons += '<span>▼</span>';
                
                html += `<th class="${{classes.join(' ')}}" onclick="window['${{tableId}}_openMenu'](event, ${{i}})">
                    <div class="th-content">
                        <span class="th-name">${{col.name}}</span>
                        <span class="th-type">${{col.type}}</span>
                        <div class="th-icons">${{icons}}</div>
                    </div>
                </th>`;
            }});
            theadRow.innerHTML = html;
        }}

        function renderBody(limit=1000) {{
            const rows = displayRows.slice(0, limit);
            let html = '';
            rows.forEach((row, rowIndex) => {{
                let tds = `<td class="f-idx-cell">${{rowIndex + 1}}</td>`;
                visibleCols.forEach(col => {{
                    let val = row[col.idx]; 
                    if(val === null) val = '<span style="color:#ccc">null</span>';
                    tds += `<td onclick="window['${{tableId}}_selectCell'](this, '${{val}}')">${{val}}</td>`;
                }});
                html += `<tr>${{tds}}</tr>`;
            }});
            tbody.innerHTML = html;
            
            let txt = `Showing ${{rows.length.toLocaleString()}} of ${{displayRows.length.toLocaleString()}} rows`;
            if(displayRows.length < fullData.length) txt += " (Filtered)";
            countSpan.innerText = txt;
        }}
        
        window[tableId + '_selectCell'] = function(el, val) {{
            const prev = tbody.querySelector('.is-selected');
            if(prev) prev.classList.remove('is-selected');
            el.parentElement.classList.add('is-selected');
            selectedCellVal = val;
        }};

        // --- STATS LOGIC (Bar Charts) ---
        window[tableId + '_stats'] = function() {{
            closeMenu();
            const dataIdx = activeColMeta.idx;
            const colName = activeColMeta.name;
            const values = displayRows.map(r => r[dataIdx]); // Use filtered data
            
            let total = values.length;
            let nulls = 0;
            let empties = 0;
            let validValues = [];
            
            values.forEach(v => {{
                if (v === null) nulls++;
                else if (v === "") empties++;
                else validValues.push(v);
            }});
            
            let normal = total - nulls;
            let unique = new Set(validValues).size;
            
            let counts = {{}};
            validValues.forEach(v => {{
                let k = String(v);
                counts[k] = (counts[k] || 0) + 1;
            }});
            
            // Sort by frequency
            let sortedCounts = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 20);
            let maxCount = sortedCounts.length > 0 ? sortedCounts[0][1] : 1;

            let barsHtml = '';
            if (sortedCounts.length === 0) {{
                barsHtml = '<div style="padding:20px; color:#999; text-align:center">No data in current filter view</div>';
            }} else {{
                barsHtml += '<div style="margin-bottom:10px; font-weight:bold; font-size:11px; color:#666">TOP VALUES</div>';
                sortedCounts.forEach(([val, count]) => {{
                    let pct = (count / maxCount) * 100;
                    barsHtml += `
                        <div class="bar-row">
                            <div class="bar-label" title="${{val}}">${{val}}</div>
                            <div class="bar-track">
                                <div class="bar-fill" style="width: ${{pct}}%"></div>
                            </div>
                            <div class="bar-val">${{count}}</div>
                        </div>
                    `;
                }});
            }}

            let modalHtml = `
                <div class="f-modal-header">
                    <span class="f-modal-title">Statistics: <span style="color:#1f6bc0">${{colName}}</span></span>
                    <span class="f-modal-close" onclick="document.getElementById('${{tableId}}_modal').style.display='none'">✕</span>
                </div>
                <div class="f-modal-body">
                    <div class="stats-left">
                        <div style="margin-bottom:15px; font-weight:bold; color:#333">DATA QUALITY</div>
                        <div class="stat-metric"><label>Normal</label><span>${{normal.toLocaleString()}}</span></div>
                        <div class="stat-metric"><label>Null</label><span>${{nulls.toLocaleString()}}</span></div>
                        <div class="stat-metric"><label>Empty</label><span>${{empties.toLocaleString()}}</span></div>
                        <div class="stat-metric" style="margin-top:15px"><label>Unique</label><span>${{unique.toLocaleString()}}</span></div>
                        <div class="stat-metric"><label>Type</label><span>${{activeColMeta.type}}</span></div>
                    </div>
                    <div class="stats-right">${{barsHtml}}</div>
                </div>
            `;
            modal.innerHTML = modalHtml;
            modal.style.display = 'block';
        }};

        // --- FILTER / PIN / SORT ---
        window[tableId + '_filter_val'] = function(type) {{
            let val = selectedCellVal; 
            if (val === null || val === undefined) {{
                val = prompt(`Value to ${{type}}?`);
                if(val === null) return;
            }}
            activeFilters[activeColMeta.idx] = {{ type: type, val: String(val) }};
            recalcRows();
            closeMenu();
        }};
        
        window[tableId + '_on_filter_input'] = function(el) {{
            const val = el.value.toLowerCase();
            if(!val) delete activeFilters[activeColMeta.idx];
            else activeFilters[activeColMeta.idx] = {{ type: 'text', val: val }};
            recalcRows();
        }};

        window[tableId + '_pin_toggle'] = function() {{
            activeColMeta.pinned = !activeColMeta.pinned;
            colMeta.sort((a, b) => {{
                if (a.pinned && !b.pinned) return -1;
                if (!a.pinned && b.pinned) return 1;
                return a.idx - b.idx; 
            }});
            const term = searchInput.value.toLowerCase();
            if(!term) visibleCols = colMeta;
            else visibleCols = colMeta.filter(c => c.name.toLowerCase().includes(term));
            renderHeader();
            renderBody();
            closeMenu();
        }};
        
        window[tableId + '_sort'] = function(dir) {{
            const dataIdx = activeColMeta.idx;
            const type = activeColMeta.type;
            displayRows.sort((a, b) => {{
                let valA = a[dataIdx];
                let valB = b[dataIdx];
                if (valA === valB) return 0;
                if (valA === null) return 1; if (valB === null) return -1;
                if (type === 'Integer' || type === 'Double') return (Number(valA) - Number(valB)) * (dir === 'asc' ? 1 : -1);
                return String(valA).localeCompare(String(valB)) * (dir === 'asc' ? 1 : -1);
            }});
            renderBody();
            closeMenu();
        }};
        
        window[tableId + '_copy'] = function() {{
             navigator.clipboard.writeText(activeColMeta.name);
             closeMenu();
        }};

        window[tableId + '_openMenu'] = function(e, visibleIndex) {{
            e.stopPropagation();
            activeColMeta = visibleCols[visibleIndex];
            const filterObj = activeFilters[activeColMeta.idx];
            const currentText = (filterObj && filterObj.type === 'text') ? filterObj.val : "";
            const pinLabel = activeColMeta.pinned ? "Unpin column" : "Pin column";
            
            let valActionHtml = '';
            if (selectedCellVal !== null) {{
                valActionHtml = `
                    <div class="f-menu-item" onclick="window['${{tableId}}_filter_val']('include')"><span class="f-menu-icon">T</span> Include only "${{selectedCellVal}}"</div>
                    <div class="f-menu-item" onclick="window['${{tableId}}_filter_val']('exclude')"><span class="f-menu-icon">X</span> Exclude "${{selectedCellVal}}"</div>
                    <div class="f-divider"></div>
                `;
            }}

            let menuHtml = `
                <div class="f-menu-item" onclick="window['${{tableId}}_pin_toggle']()">
                    <span class="f-menu-icon">📌</span> ${{pinLabel}}
                </div>
                <div class="f-divider"></div>
                ${{valActionHtml}}
                <div class="f-menu-input-wrap">
                    <input class="f-menu-input" placeholder="Filter (contains)..." value="${{currentText}}" 
                           onkeyup="window['${{tableId}}_on_filter_input'](this)" onclick="event.stopPropagation()" autofocus>
                </div>
                <div class="f-menu-item" onclick="window['${{tableId}}_sort']('asc')"><span class="f-menu-icon">↑</span> Sort Ascending</div>
                <div class="f-menu-item" onclick="window['${{tableId}}_sort']('desc')"><span class="f-menu-icon">↓</span> Sort Descending</div>
                <div class="f-divider"></div>
                <div class="f-menu-item" onclick="window['${{tableId}}_stats']()"><span class="f-menu-icon">☰</span> View Stats</div>
                <div class="f-menu-item" onclick="window['${{tableId}}_copy']()"><span class="f-menu-icon">❐</span> Copy Name</div>
            `;
            menu.innerHTML = menuHtml;
            
            const rect = e.currentTarget.getBoundingClientRect();
            const wrapRect = wrapper.getBoundingClientRect();
            let left = rect.left - wrapRect.left;
            if(left + 220 > wrapRect.width) left = wrapRect.width - 230;
            menu.style.top = (rect.bottom - wrapRect.top) + 'px';
            menu.style.left = left + 'px';
            menu.style.display = 'block';
            setTimeout(() => {{ const inp = menu.querySelector('input'); if(inp) inp.focus(); }}, 50);
        }};

        function closeMenu() {{ menu.style.display = 'none'; }}
        document.addEventListener('click', closeMenu);
        menu.addEventListener('click', e => e.stopPropagation());
        searchInput.addEventListener('keyup', (e) => {{ 
            const term = searchInput.value.toLowerCase();
            if(!term) visibleCols = colMeta;
            else visibleCols = colMeta.filter(c => c.name.toLowerCase().includes(term));
            renderHeader();
            renderBody();
        }});
        
        renderHeader();
        renderBody(); 
    }})();
    </script>
    """

    # --- 4. HTML ASSEMBLY ---
    html = f"""
    {style}
    <div id="{table_id}_wrapper">
        <div class="f-toolbar"><div class="f-tab">⊞ Preview</div></div>
        <div class="f-statusbar">
            <span id="{table_id}_count">{showing_text}</span>
            <span style="border-left:1px solid #d1d5db; padding-left:10px;">{total_cols} cols</span>
            <div class="f-search"><input id="{table_id}_search" type="text" placeholder="Search columns..."></div>
        </div>
        <div class="f-scroll">
            <table id="{table_id}_table" class="f-table">
                <thead><tr id="{table_id}_thead_row"></tr></thead>
                <tbody id="{table_id}_tbody"></tbody>
            </table>
        </div>
        <div id="{table_id}_menu" class="f-menu"></div>
        <div id="{table_id}_modal" class="f-modal"></div>
    </div>
    {script}
    """
    display(HTML(html))