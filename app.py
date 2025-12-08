import streamlit as st
import sqlite3
import pandas as pd
import uuid
import json
import streamlit.components.v1 as components

# --- 1. Page Config ---
st.set_page_config(
    page_title="Dataset Preview",
    page_icon="🗃️",
    layout="wide"
)

# --- 2. Custom CSS ---
st.markdown("""
    <style>
        .breadcrumb { color: #5f6b7c; font-size: 0.9rem; margin-bottom: 10px; }
        .current-path { font-weight: bold; color: #10161a; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] { height: 50px; }
        textarea { font-family: 'Source Code Pro', monospace !important; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Database Connection ---
@st.cache_resource
def get_connection():
    return sqlite3.connect("market_data.db", check_same_thread=False)

conn = get_connection()

# ==========================================
# 4. FOUNDRY PRO DISPLAY COMPONENT
# ==========================================
def foundry_pro_display(df: pd.DataFrame, limit_visual_rows: int = 50000):
    """
    Renders DataFrame with Foundry-like visuals using HTML/JS Injection.
    - Added: Drag-to-resize columns.
    """
    table_id = f"foundry_{uuid.uuid4().hex[:8]}"
    
    # --- Data Serialization ---
    df_safe = df.fillna("") 
    
    def get_friendly_type(dtype):
        s = str(dtype)
        if 'int' in s: return 'Integer'
        if 'float' in s: return 'Double'
        if 'date' in s: return 'Date'
        if 'bool' in s: return 'Boolean'
        return 'String'

    cols_config = [{
        'name': str(c), 
        'type': get_friendly_type(df[c].dtype), 
        'idx': i,      
        'pinned': False,
        'expanded': False,
        'width': 150 # Default width
    } for i, c in enumerate(df.columns)]
    
    try:
        json_data = df_safe.to_json(orient='values', date_format='iso')
    except:
        json_data = json.dumps(df_safe.values.tolist(), default=str)
    
    json_cols = json.dumps(cols_config)
    total_rows = len(df)
    total_cols = len(df.columns)
    
    showing_text = f"Showing top {limit_visual_rows:,} of {total_rows:,} rows" if total_rows > limit_visual_rows else f"Showing all {total_rows:,} rows"

    # --- CSS STYLING ---
    style = f"""
    <style>
        #{table_id}_wrapper {{
            font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            border: 1px solid #d1d5db; background: #fff; font-size: 12px; margin-bottom: 20px;
            position: relative; user-select: none; display: flex; flex-direction: column; height: 600px;
        }}
        .f-toolbar {{ border-bottom: 1px solid #e5e7eb; padding: 0 10px; height: 35px; min-height: 35px; display: flex; align-items: center; background: #fff; }}
        .f-tab {{ color: #1f6bc0; font-weight: 600; border-bottom: 2px solid #1f6bc0; height: 100%; display: flex; align-items: center; padding: 0 12px; cursor: pointer; }}
        .f-statusbar {{ background: #f7f8fa; border-bottom: 1px solid #d1d5db; padding: 4px 12px; display: flex; justify-content: flex-end; align-items: center; gap: 15px; color: #5f6b7c; font-size: 11px; height: 30px; min-height: 30px; }}
        .f-search input {{ border: 1px solid #d1d5db; padding: 3px 8px 3px 22px; font-size: 11px; width: 180px; border-radius: 3px; outline: none; }}
        .f-search {{ position: relative; }}
        .f-search:before {{ content: "🔍"; position: absolute; left: 6px; top: 3px; opacity: 0.5; font-size:10px; }}
        
        /* Table Scroll Area */
        .f-scroll {{ overflow: auto; flex: 1; position: relative; background: #fff; }}
        table.f-table {{ border-collapse: separate; border-spacing: 0; min-width: 100%; table-layout: fixed; }}
        
        .f-table thead th {{
            position: sticky; top: 0; z-index: 20; background: #f7f8fa;
            border-bottom: 1px solid #d1d5db; border-right: 1px solid #d1d5db;
            padding: 0; height: 40px; text-align: left; vertical-align: top; cursor: pointer;
            /* Width controlled by JS now */
        }}
        .f-table thead th:hover {{ background: #eff6ff; }}
        .f-table thead th.is-pinned {{ background: #f0f6ff; border-bottom: 2px solid #1f6bc0; }} 
        .f-table thead th.is-filtered .th-name {{ color: #1f6bc0; }}
        
        .th-content {{ padding: 5px 8px; display: flex; flex-direction: column; justify-content: center; height: 100%; position: relative; overflow: hidden; }}
        .th-name {{ font-weight: 700; color: #10161a; margin-bottom: 2px; font-size: 11px; }}
        .th-type {{ font-size: 9px; color: #8a9ba8; font-family: monospace; text-transform: uppercase; }}
        .th-icons {{ position: absolute; top: 4px; right: 4px; font-size: 10px; color: #1f6bc0; display: flex; gap: 4px; }}
        
        /* RESIZER HANDLE */
        .f-resizer {{
            position: absolute; right: 0; top: 0; width: 5px; height: 100%; cursor: col-resize; z-index: 30;
            background: transparent;
        }}
        .f-resizer:hover {{ background: #1f6bc0; opacity: 0.5; }}

        .f-idx-head {{ width: 40px; min-width: 40px; z-index: 25 !important; left: 0; border-right: 2px solid #d1d5db !important; }}
        .f-idx-cell {{ position: sticky; left: 0; z-index: 10; background: #f7f8fa; border-right: 2px solid #d1d5db; text-align: right; color: #9aa5b1; font-weight: 600; font-size: 10px; }}
        
        .f-table tbody td {{
            padding: 4px 8px; border-bottom: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #333; font-size: 11px;
        }}
        .f-table tbody tr:hover td {{ background: #f5f8fa; }}
        .f-table tbody tr.is-selected td {{ background: #e5efff; }}

        /* Context Menu */
        .f-menu {{
            display: none; position: absolute; background: white; border: 1px solid #c5cbd3;
            box-shadow: 0 0 0 1px rgba(16,22,26,.1), 0 4px 8px rgba(16,22,26,.2); 
            border-radius: 3px; z-index: 99999; min-width: 180px; padding: 4px 0; font-family: sans-serif;
        }}
        .f-menu-item {{ padding: 6px 12px; cursor: pointer; display: flex; align-items: center; color: #10161a; font-size: 12px; }}
        .f-menu-item:hover {{ background: #1f6bc0; color: white; }}
        .f-menu-item:hover .f-menu-icon {{ color: white; }}
        .f-menu-icon {{ width: 20px; text-align: left; margin-right: 5px; color: #5f6b7c; font-size: 12px; }}
        .f-divider {{ height: 1px; background: #e5e7eb; margin: 3px 0; }}
        .f-menu-input-wrap {{ padding: 6px 12px; border-bottom: 1px solid #eee; background: #fafafa; }}
        .f-menu-input {{ width: 100%; box-sizing: border-box; padding: 4px; border: 1px solid #ccc; border-radius: 3px; font-size: 11px; outline: none; }}
        
        /* Stats Panel */
        .f-stats-panel {{
            display: none; height: 250px; border-top: 1px solid #d1d5db; background: #fff;
            flex-direction: column; z-index: 50; box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        }}
        .stats-header {{ height: 30px; background: #f7f8fa; border-bottom: 1px solid #d1d5db; display: flex; align-items: center; justify-content: space-between; padding: 0 10px; }}
        .stats-title {{ font-weight: 700; color: #333; font-size: 11px; text-transform: uppercase; }}
        .stats-close {{ cursor: pointer; color: #5f6b7c; font-size: 14px; font-weight: bold; }}
        .stats-content {{ display: flex; height: 220px; }}
        .stats-metrics {{ width: 220px; padding: 15px; border-right: 1px solid #e5e7eb; overflow-y: auto; }}
        .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 11px; }}
        .metric-label {{ color: #5f6b7c; }}
        .metric-val {{ font-weight: 600; color: #10161a; }}
        .metric-bar-bg {{ height: 4px; background: #eee; width: 100%; margin-top: 2px; border-radius: 2px; }}
        .metric-bar-fill {{ height: 100%; background: #5f6b7c; border-radius: 2px; }}
        .stats-charts {{ flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; }}
        .chart-row {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 11px; height: 16px; }}
        .chart-label {{ width: 140px; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #10161a; padding-right: 10px; }}
        .chart-val {{ width: 50px; text-align: right; color: #10161a; font-weight: 600; padding-right: 10px; }}
        .chart-track {{ flex: 1; height: 10px; background: #f1f5f9; border-radius: 2px; position: relative; }}
        .chart-fill {{ height: 100%; background: #9cb4cd; border-radius: 2px; }}
    </style>
    """

    # --- JAVASCRIPT LOGIC ---
    script = f"""
    <script>
    (function() {{
        const tableId = "{table_id}";
        const wrapper = document.getElementById(tableId + '_wrapper');
        const tbody = document.getElementById(tableId + '_tbody');
        const theadRow = document.getElementById(tableId + '_thead_row');
        const countSpan = document.getElementById(tableId + '_count');
        const menu = document.getElementById(tableId + '_menu');
        const statsPanel = document.getElementById(tableId + '_stats_panel');
        const statsTitle = document.getElementById(tableId + '_stats_title');
        const statsMetrics = document.getElementById(tableId + '_stats_metrics');
        const statsCharts = document.getElementById(tableId + '_stats_charts');
        
        let fullData = {json_data}; 
        let colMeta = {json_cols}; 
        let activeFilters = {{}}; 
        let visibleCols = colMeta; 
        let displayRows = fullData; 
        let activeColMeta = null;
        let selectedCellVal = null;
        
        // RESIZING VARIABLES
        let isResizing = false;
        let currentResizeCol = null;
        let startX = 0;
        let startWidth = 0;

        // --- 1. RENDER LOGIC ---
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
                
                // Use stored width
                let widthStyle = `width: ${{col.width}}px`;
                
                html += `<th class="${{classes.join(' ')}}" style="${{widthStyle}}" onclick="window['${{tableId}}_openMenu'](event, ${{i}})">
                    <div class="th-content">
                        <div style="display:flex; justify-content:space-between">
                            <span class="th-name">${{col.name}}</span>
                            <div class="th-icons">${{icons}}</div>
                        </div>
                        <span class="th-type">${{col.type}}</span>
                    </div>
                    <div class="f-resizer" onmousedown="window['${{tableId}}_initResize'](event, ${{i}})" onclick="event.stopPropagation()"></div>
                </th>`;
            }});
            theadRow.innerHTML = html;
        }}

        function renderBody(limit=500) {{
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

        // --- 2. RESIZE LOGIC (New!) ---
        window[tableId + '_initResize'] = function(e, colIndex) {{
            e.stopPropagation(); // Don't trigger sort/menu
            isResizing = true;
            currentResizeCol = visibleCols[colIndex];
            startX = e.pageX;
            startWidth = currentResizeCol.width;
            
            document.body.style.cursor = 'col-resize';
            // Add listeners to document to handle drag outside the header
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        }};
        
        function onMouseMove(e) {{
            if (!isResizing) return;
            const deltaX = e.pageX - startX;
            let newWidth = startWidth + deltaX;
            if(newWidth < 50) newWidth = 50; // Min width
            currentResizeCol.width = newWidth;
            
            // Re-render header to update width immediately
            renderHeader();
        }}
        
        function onMouseUp(e) {{
            isResizing = false;
            document.body.style.cursor = 'default';
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }}

        // --- 3. MENU & STATS ACTIONS ---
        window[tableId + '_stats'] = function() {{
            closeMenu();
            const dataIdx = activeColMeta.idx;
            const colName = activeColMeta.name;
            const colType = activeColMeta.type;
            const values = displayRows.map(r => r[dataIdx]);
            
            let total = values.length;
            let nulls = 0; let empties = 0; let validValues = [];
            values.forEach(v => {{
                if (v === null) nulls++;
                else if (v === "") empties++;
                else validValues.push(v);
            }});
            
            let normal = total - nulls;
            let unique = new Set(validValues).size;
            let counts = {{}};
            validValues.forEach(v => {{ counts[String(v)] = (counts[String(v)] || 0) + 1; }});
            let sortedCounts = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 20);
            let maxCount = sortedCounts.length > 0 ? sortedCounts[0][1] : 1;

            statsTitle.innerText = colName + " (" + colType + ")";
            statsMetrics.innerHTML = `
                <div class="metric-row"><span class="metric-label">Normal</span><span class="metric-val">${{normal.toLocaleString()}}</span></div>
                <div class="metric-row" style="margin-bottom:15px"><div class="metric-bar-bg"><div class="metric-bar-fill" style="width:${{(normal/total)*100}}%"></div></div></div>
                <div class="metric-row"><span class="metric-label">Null</span><span class="metric-val">${{nulls.toLocaleString()}}</span></div>
                <div class="metric-row"><span class="metric-label">Unique</span><span class="metric-val">${{unique.toLocaleString()}}</span></div>
            `;

            let chartsHtml = '';
            if (sortedCounts.length === 0) {{ chartsHtml = '<div style="color:#999; padding:20px;">No data</div>'; }} 
            else {{
                sortedCounts.forEach(([val, count]) => {{
                    let pct = (count / maxCount) * 100;
                    chartsHtml += `<div class="chart-row">
                        <div class="chart-val">${{count}}</div>
                        <div class="chart-track"><div class="chart-fill" style="width: ${{pct}}%"></div></div>
                        <div class="chart-label" style="padding-left:10px" title="${{val}}">${{val}}</div>
                    </div>`;
                }});
            }}
            statsCharts.innerHTML = chartsHtml;
            statsPanel.style.display = 'flex';
        }};

        window[tableId + '_closeStats'] = function() {{ statsPanel.style.display = 'none'; }};

        window[tableId + '_filter_val'] = function(type) {{
            let val = selectedCellVal || prompt(`Value to ${{type}}?`);
            if(val === null) return;
            activeFilters[activeColMeta.idx] = {{ type: type, val: String(val) }};
            recalcRows(); closeMenu();
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
            visibleCols = colMeta; renderHeader(); renderBody(); closeMenu();
        }};

        window[tableId + '_expand_toggle'] = function() {{
            // Manual Expand sets width to 400
            activeColMeta.width = activeColMeta.width > 300 ? 150 : 400;
            renderHeader(); closeMenu();
        }};

        window[tableId + '_sort'] = function(dir) {{
            const dataIdx = activeColMeta.idx;
            displayRows.sort((a, b) => {{
                let valA = a[dataIdx]; let valB = b[dataIdx];
                if (valA === valB) return 0;
                if (valA === null) return 1; if (valB === null) return -1;
                return String(valA).localeCompare(String(valB)) * (dir === 'asc' ? 1 : -1);
            }});
            renderBody(); closeMenu();
        }};
        
        window[tableId + '_openMenu'] = function(e, visibleIndex) {{
            e.stopPropagation();
            activeColMeta = visibleCols[visibleIndex];
            const filterObj = activeFilters[activeColMeta.idx];
            const currentText = (filterObj && filterObj.type === 'text') ? filterObj.val : "";
            
            // Dynamic Label
            const expandLabel = activeColMeta.width > 300 ? "Collapse column" : "Expand column";
            const expandIcon = activeColMeta.width > 300 ? "⇤" : "⇥";

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
                    <span class="f-menu-icon">📌</span> ${{activeColMeta.pinned ? "Unpin" : "Pin"}} column
                </div>
                <div class="f-menu-item" onclick="window['${{tableId}}_expand_toggle']()">
                    <span class="f-menu-icon">${{expandIcon}}</span> ${{expandLabel}}
                </div>
                <div class="f-divider"></div>
                ${{valActionHtml}}
                <div class="f-menu-input-wrap">
                    <input class="f-menu-input" placeholder="Filter..." value="${{currentText}}" 
                           onkeyup="window['${{tableId}}_on_filter_input'](this)" onclick="event.stopPropagation()" autofocus>
                </div>
                <div class="f-menu-item" onclick="window['${{tableId}}_sort']('asc')"><span class="f-menu-icon">↑</span> Sort Ascending</div>
                <div class="f-menu-item" onclick="window['${{tableId}}_sort']('desc')"><span class="f-menu-icon">↓</span> Sort Descending</div>
                <div class="f-divider"></div>
                <div class="f-menu-item" onclick="window['${{tableId}}_stats']()"><span class="f-menu-icon">☰</span> View Stats</div>
            `;
            menu.innerHTML = menuHtml;
            
            const rect = e.currentTarget.getBoundingClientRect();
            const wrapRect = wrapper.getBoundingClientRect();
            let left = rect.left - wrapRect.left;
            if(left + 200 > wrapRect.width) left = wrapRect.width - 200;
            menu.style.top = (rect.bottom - wrapRect.top) + 'px';
            menu.style.left = left + 'px';
            menu.style.display = 'block';
            setTimeout(() => {{ const inp = menu.querySelector('input'); if(inp) inp.focus(); }}, 50);
        }};

        function closeMenu() {{ menu.style.display = 'none'; }}
        document.addEventListener('click', closeMenu);
        menu.addEventListener('click', e => e.stopPropagation());
        
        renderHeader();
        renderBody(); 
    }})();
    </script>
    """

    # --- HTML ASSEMBLY ---
    html = f"""
    {style}
    <div id="{table_id}_wrapper">
        <div class="f-toolbar">
            <div class="f-tab">⊞ Preview</div>
        </div>
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
        
        <div id="{table_id}_stats_panel" class="f-stats-panel">
            <div class="stats-header">
                <span class="stats-title" id="{table_id}_stats_title">COLUMN STATS</span>
                <span class="stats-close" onclick="window['{table_id}_closeStats']()">✕</span>
            </div>
            <div class="stats-content">
                <div id="{table_id}_stats_metrics" class="stats-metrics"></div>
                <div id="{table_id}_stats_charts" class="stats-charts"></div>
            </div>
        </div>

        <div id="{table_id}_menu" class="f-menu"></div>
    </div>
    {script}
    """
    components.html(html, height=650, scrolling=False)


# ==========================================
# 5. MAIN APP UI
# ==========================================

# --- Sidebar (Metadata) ---
with st.sidebar:
    st.header("🗃️ Dataset Selector")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [x[0] for x in cursor.fetchall()]
    except:
        tables = []

    selected_table = st.selectbox("Choose Dataset", tables) if tables else "No Data"
    st.divider()
    st.subheader(f"About: {selected_table}")
    st.markdown(f"""
    **Type:** Dataset  
    **Location:** `/Palantir/Data/Market/{selected_table}`  
    **RID:** `{str(uuid.uuid4())}`  
    **Updated:** Today, 12:45 PM by *User* **Created:** Nov 20, 2025 by *Gemini*
    """)
    st.info("ℹ️ **New Feature:** You can now drag the right edge of any column header to resize it.")

# --- Header ---
st.markdown(f"""
<div class="breadcrumb">
    Palantir / Data / Market / <span class="current-path">{selected_table}</span>
</div>
""", unsafe_allow_html=True)

st.title(f"📄 {selected_table}")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Preview", "History", "Details", "Health"])

with tab1:
    # --- Toolbar & SQL Editor ---
    col1, col2 = st.columns([0.85, 0.15])
    default_query = f"SELECT * FROM {selected_table} LIMIT 5000"
    
    with col1:
        with st.expander("📝 SQL Editor (Editable)", expanded=False):
            active_query = st.text_area(
                "Edit Query", 
                value=default_query, 
                height=150,
                key=f"sql_{selected_table}" 
            )
            st.caption("Press Ctrl+Enter to apply changes.")

    with col2:
        st.button("📊 Analyze", use_container_width=True)

    # --- Data Display ---
    if selected_table != "No Data":
        try:
            df = pd.read_sql(active_query, conn)
            foundry_pro_display(df)
        except Exception as e:
            st.error(f"⚠️ SQL Error: {e}")

with tab2:
    st.write("### 📜 Build History")
    st.info("No previous builds found.")

with tab3:
    st.write("### 🔍 Column Details")
    if selected_table != "No Data" and 'df' in locals():
        st.table(df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Type"}))

with tab4:
    st.success("✅ Dataset Health: Healthy")