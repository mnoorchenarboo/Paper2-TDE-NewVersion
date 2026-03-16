import pandas as pd
from itables import init_notebook_mode, show
import itables.options as opt
from IPython.display import HTML

# Style to center-align table content
HTML("""
<style>
td.dt-center, th.dt-center {
    text-align: center !important;
}
</style>
""")

# Enable interactive notebook mode
init_notebook_mode(all_interactive=True)

# Set constant export buttons (these are always useful)
opt.buttons = ["copyHtml5", "csvHtml5", "excelHtml5"]

# Function to display DataFrame, with optional downsampling
def myshow(df, downsample=False):
    if downsample:
        # Default or safe limits to avoid memory issues
        opt.maxBytes = 250_000
        opt.maxRows = 1_000
        opt.maxColumns = 200
        opt.default_length = 10
    else:
        # Disable all downsampling
        opt.maxBytes = 0
        opt.maxRows = 1_000_000
        opt.maxColumns = 0
        opt.default_length = -1

    # Get type of the first column for searchBuilder filter
    first_col = df.columns[0]
    first_val = df.iloc[0, 0]

    if pd.api.types.is_numeric_dtype(df[first_col]):
        val_type = "num"
    elif pd.api.types.is_datetime64_any_dtype(df[first_col]):
        val_type = "date"
    else:
        val_type = "string"

    # Fix JSON serialization by converting NumPy scalars to Python types
    if hasattr(first_val, "item"):
        first_val = first_val.item()

    val = [first_val] if val_type != "num" else first_val

    show(
        df,
        layout={"top1": "searchBuilder"},
        style="margin-left: auto; margin-right: auto; table-layout: auto; width: 80%;",
        searchBuilder={
            "preDefined": {
                "criteria": [
                    {
                        "data": first_col,
                        "condition": "=",
                        "value": val,
                        "type": val_type
                    }
                ]
            }
        }
    )
