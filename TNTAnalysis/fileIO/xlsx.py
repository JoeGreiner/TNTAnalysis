import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell


def write_trackmate_xlsx(df_spot, df_edge, df_track, path_xlsx_out, color_numeric=True):
    with pd.ExcelWriter(path_xlsx_out, engine='xlsxwriter') as writer:
        for sheet_name, df in (('spots', df_spot), ('edges', df_edge), ('tracks', df_track)):
            index_columns = df.index.nlevels
            df = df.reset_index()
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            format_excel_sheet(df, writer, sheet_name, color_numeric, index_columns)


def is_numeric_column(column):
    column = column.dropna()
    column = column[column != '']
    return pd.to_numeric(column, errors='coerce').notna().all()


def format_excel_sheet(df, writer, sheet_name, color_numeric=True, index_columns=1):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#000000',
        'font_color': '#FFFFFF',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
    })
    for col_num, column in enumerate(df.columns):
        worksheet.write_string(0, col_num, str(column), header_format)
    worksheet.autofit()
    worksheet.freeze_panes(1, index_columns)

    if not color_numeric:
        return
    constant_format = workbook.add_format({'bg_color': '#FFFF00'})
    for col_num, column in enumerate(df.columns):
        if pd.api.types.is_bool_dtype(df[column]) or not is_numeric_column(df[column]):
            continue
        values = pd.to_numeric(df[column], errors='coerce').dropna()
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        if values.nunique() == 1:
            worksheet.conditional_format(1, col_num, len(df), col_num, {
                'type': 'formula',
                'criteria': f'=ISNUMBER({xl_rowcol_to_cell(1, col_num)})',
                'format': constant_format,
            })
            continue
        worksheet.conditional_format(1, col_num, len(df), col_num, {
            'type': '3_color_scale',
            'min_type': 'num',
            'min_value': values.min(),
            'min_color': '#FF0000',
            'mid_type': 'num',
            'mid_value': values.median(),
            'mid_color': '#FFFF00',
            'max_type': 'num',
            'max_value': values.max(),
            'max_color': '#00FF00',
        })
