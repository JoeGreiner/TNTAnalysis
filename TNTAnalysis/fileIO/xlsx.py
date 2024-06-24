import numpy as np
import pandas as pd


def write_trackmate_xlsx(df_spot, df_edge, df_track, path_xlsx_out):
    df_spot = df_spot.reset_index()
    df_edge = df_edge.reset_index()
    df_track = df_track.reset_index()
    writer = pd.ExcelWriter(path_xlsx_out)
    df_spot.to_excel(writer, sheet_name='spots', index=False, freeze_panes=(1, 1))
    df_edge.to_excel(writer, sheet_name='edges', index=False, freeze_panes=(1, 1))
    df_track.to_excel(writer, sheet_name='tracks', index=False, freeze_panes=(1, 1))
    fix_df_lengths(df_spot, writer, 'spots')
    fix_df_lengths(df_edge, writer, 'edges')
    fix_df_lengths(df_track, writer, 'tracks')
    writer.close()

def fix_df_lengths(df, writer, sheetName):
    for column in df:
        column_name_length = 1.4*len(column) # can be bigger because of bold/capital letters
        column_entry_max_lengths = df[column].astype(str).map(len).max()
        column_length = max(column_entry_max_lengths, column_name_length)
        col_idx = df.columns.get_loc(column)
        if np.isnan(column_length):
            column_length = 1
        writer.sheets[sheetName].set_column(col_idx, col_idx, int(column_length))