from IPython.display import display, HTML
import json
import uuid
import pandas as pd # Import pandas for the function argument

def interactive_dataframe_display(df: pd.DataFrame, title="DataFrame Analysis"):
    """
    Creates an interactive HTML table with sorting, filtering, and 
    a 'Download as CSV' button for the filtered data.
    
    Args:
        df: Pandas DataFrame to display
        title: Title for the display
        
    Returns:
        HTML object that can be displayed in a notebook
    """
    # Convert dataframe to JSON for JavaScript processing
    # Using 'records' orientation for easy row-by-row iteration in JS
    json_data = df.to_json(orient='records')
    
    # Generate a unique ID for this table instance to avoid conflicts
    table_id = f"table_{uuid.uuid4().hex[:8]}"
    
    # Get column names for CSV header
    columns = list(df.columns)
    column_names_js = json.dumps(columns)
    
    # Create HTML with embedded JavaScript for interactivity
    html_content = f"""
    <div style="padding: 10px; border: 2px solid #4287f5; border-radius: 5px; margin-bottom: 20px; background-color: #121212; color: white;">
        <h2 style="color: #4287f5;">{title}</h2>
        <p>This dataset contains <span id="{table_id}_row_count">{len(df)}</span> rows (showing <span id="{table_id}_visible_count">{len(df)}</span>).</p>
        
        <div style="margin: 10px 0; display: flex; gap: 10px;">
            <button id="{table_id}_reset_filters" style="padding: 5px 10px; background-color: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer;">Reset All Filters</button>
            <button id="{table_id}_download_csv" style="padding: 5px 10px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">Download as CSV</button>
        </div>
        
        <div style="overflow-x: auto; max-height: 500px; overflow-y: auto;">
            <table id="{table_id}" style="width: 100%; border-collapse: collapse; margin-top: 10px; background-color: #121212; color: white;">
                <thead>
                    <tr>
                        {
                            ''.join([f'''
                            <th style="position: sticky; top: 0; background-color: #000000; color: white; padding: 8px; text-align: left;" data-column="{col}" data-index="{i}">
                                <div style="margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                                    <span>{col}</span>
                                    <span class="{table_id}_sort_icon" style="cursor: pointer;">⇅</span>
                                </div>
                                <input type="text" class="{table_id}_filter_input" data-column="{col}" placeholder="Filter..." 
                                            style="width: 100%; padding: 3px; border-radius: 3px; border: 1px solid #444; background-color: #2a2a2a; color: white; font-size: 0.9em;">
                            </th>
                            ''' for i, col in enumerate(columns)])
                        }
                    </tr>
                </thead>
                <tbody>
                    </tbody>
            </table>
        </div>
    </div>

    <script>
        (function() {{
            // Store data and table references
            const tableId = "{table_id}";
            const tableData = {json_data};
            const columns = {column_names_js};
            
            // Create a namespace for this table instance
            window[tableId] = {{
                data: tableData,
                filteredData: [...tableData],
                sortColumn: null,
                sortDirection: 1
            }};
            
            // --- Utility Functions ---

            // Function to sanitize data for CSV
            function csvSanitize(value) {{
                // Handle null, undefined, or empty values
                if (value === null || value === undefined) return "";

                // Convert to string and escape double quotes
                let strValue = String(value).replace(/"/g, '""');

                // Enclose in double quotes if it contains commas, double quotes, or newlines
                if (strValue.includes(',') || strValue.includes('\\n') || strValue.includes('"')) {{
                    return `"${{strValue}}"`;
                }}
                return strValue;
            }}

            // Function to render the table
            function renderTable() {{
                const tbody = document.querySelector(`#${{tableId}} tbody`);
                tbody.innerHTML = '';
                
                window[tableId].filteredData.forEach((row, rowIndex) => {{
                    const tr = document.createElement('tr');
                    tr.style.backgroundColor = rowIndex % 2 === 0 ? '#121212' : '#1a1a1a';
                    tr.style.color = 'white';
                    
                    columns.forEach(col => {{
                        const td = document.createElement('td');
                        td.style.padding = '8px';
                        td.style.borderBottom = '1px solid #333';
                        td.textContent = row[col] !== null && row[col] !== undefined ? row[col] : '';
                        tr.appendChild(td);
                    }});
                    
                    // Add hover effect
                    tr.addEventListener('mouseover', () => {{
                        tr.style.backgroundColor = '#2a2a2a';
                    }});
                    tr.addEventListener('mouseout', () => {{
                        tr.style.backgroundColor = rowIndex % 2 === 0 ? '#121212' : '#1a1a1a';
                    }});
                    
                    tbody.appendChild(tr);
                }});
                
                // Update row counts
                document.getElementById(`${{tableId}}_visible_count`).textContent = window[tableId].filteredData.length;
            }}
            
            // Sort the table
            function sortTable(column, forceDirection = null) {{
                // Toggle sort direction if clicking the same column
                if (window[tableId].sortColumn === column && forceDirection === null) {{
                    window[tableId].sortDirection *= -1;
                }} else {{
                    window[tableId].sortColumn = column;
                    window[tableId].sortDirection = forceDirection !== null ? forceDirection : 1;
                }}
                
                // Reset all sort icons
                document.querySelectorAll(`.${{tableId}}_sort_icon`).forEach(icon => {{
                    icon.textContent = '⇅';
                }});
                
                // Find the header for this column and update its sort icon
                const headers = document.querySelectorAll(`#${{tableId}} th`);
                let headerElement = null;
                
                for (let i = 0; i < headers.length; i++) {{
                    if (headers[i].dataset.column === column) {{
                        headerElement = headers[i];
                        break;
                    }}
                }}
                
                if (headerElement) {{
                    const sortIcon = headerElement.querySelector(`.${{tableId}}_sort_icon`);
                    if (sortIcon) {{
                        sortIcon.textContent = window[tableId].sortDirection === 1 ? '↑' : '↓';
                    }}
                }}
                
                // Sort the data
                window[tableId].filteredData.sort((a, b) => {{
                    const valueA = a[column];
                    const valueB = b[column];
                    
                    // Handle null/undefined values
                    if (valueA === null || valueA === undefined) return window[tableId].sortDirection;
                    if (valueB === null || valueB === undefined) return -window[tableId].sortDirection;
                    
                    // Compare based on data type
                    if (typeof valueA === 'number' && typeof valueB === 'number') {{
                        return (valueA - valueB) * window[tableId].sortDirection;
                    }} else {{
                        // Convert to string for comparison
                        return String(valueA).localeCompare(String(valueB)) * window[tableId].sortDirection;
                    }}
                }});
                
                renderTable();
            }}
            
            // Apply filters
            function applyFilters() {{
                const filterInputs = document.querySelectorAll(`.${{tableId}}_filter_input`);
                
                // 1. Filter the data from the original dataset
                window[tableId].filteredData = window[tableId].data.filter(row => {{
                    return Array.from(filterInputs).every(input => {{
                        const column = input.dataset.column;
                        const filterValue = input.value.toLowerCase();
                        
                        if (!filterValue) return true; // Skip empty filters
                        
                        const cellValue = row[column];
                        if (cellValue === null || cellValue === undefined) return false;
                        
                        return String(cellValue).toLowerCase().includes(filterValue);
                    }});
                }});
                
                // 2. Re-apply current sort if one exists
                if (window[tableId].sortColumn) {{
                    // The sortTable function will sort the filteredData and then call renderTable
                    // We need to pass the current direction as forceDirection to prevent it from toggling
                    sortTable(window[tableId].sortColumn, window[tableId].sortDirection); 
                }} else {{
                    // 3. Render the newly filtered, unsorted data
                    renderTable();
                }}
            }}
            
            // Reset filters
            function resetFilters() {{
                document.querySelectorAll(`.${{tableId}}_filter_input`).forEach(input => {{
                    input.value = '';
                }});
                window[tableId].filteredData = [...window[tableId].data];
                
                // Reset sort indicators
                document.querySelectorAll(`.${{tableId}}_sort_icon`).forEach(icon => {{
                    icon.textContent = '⇅';
                }});
                window[tableId].sortColumn = null;
                window[tableId].sortDirection = 1;
                
                renderTable();
            }}
            
            // Function to handle CSV download
            function downloadCSV() {{
                const data = window[tableId].filteredData;
                if (data.length === 0) {{
                    alert("No visible data to download.");
                    return;
                }}

                // 1. Create CSV Header
                const header = columns.map(csvSanitize).join(',');

                // 2. Create CSV Body
                const rows = data.map(row => {{
                    return columns.map(col => csvSanitize(row[col])).join(',');
                }});

                const csvContent = [header, ...rows].join('\\n');

                // 3. Create Blob and Download Link
                const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                const link = document.createElement("a");
                const url = URL.createObjectURL(blob);
                
                link.setAttribute("href", url);
                link.setAttribute("download", `filtered_data_${{tableId}}.csv`);
                link.style.visibility = 'hidden';
                
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}

            // --- Event Listeners ---
            
            // Initialize the table
            renderTable();
            
            // Add event listeners for sorting icons
            document.querySelectorAll(`.${{tableId}}_sort_icon`).forEach(icon => {{
                icon.addEventListener('click', function(event) {{
                    event.stopPropagation();
                    const column = this.closest('th').dataset.column;
                    sortTable(column);
                }});
            }});
            
            // Add event listeners for filtering inputs
            document.querySelectorAll(`.${{tableId}}_filter_input`).forEach(input => {{
                input.addEventListener('keyup', function(event) {{
                    applyFilters();
                    if (event.key === 'Enter') {{
                        this.blur();
                    }}
                    event.stopPropagation();
                }});
                
                input.addEventListener('click', function(event) {{
                    event.stopPropagation(); // Prevent filter click from triggering sort
                }});
            }});
            
            // Add event listener for reset button
            document.getElementById(`${{tableId}}_reset_filters`).addEventListener('click', resetFilters);
            
            // Add event listener for download button
            document.getElementById(`${{tableId}}_download_csv`).addEventListener('click', downloadCSV);
        }})();
    </script>
    """
    
    return HTML(html_content)