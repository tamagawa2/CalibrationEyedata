import numpy as np
import gradio as gr
import numpy as np
from PIL import Image
import io

import CaliCoodinate
import CaliOffset


calico = CaliCoodinate.CaliCoodinate()
caliof = CaliOffset.CaliOffset()

# Blocksの作成
with gr.Blocks() as demo:
    # UI
    gr.Markdown("目の動きの調整をします")
    with gr.Tabs():
        # CaliCoodinateタブ
        with gr.TabItem("CaliCoodinate", interactive=True):
            calico_path = gr.Textbox(label="CaliCoodinateのフォルダのパス")
            calico_outoutpath = gr.Textbox(label="CaliCoodinateの出力フォルダのパス")
            batchsize_co = gr.Number(label="batchsize")
            num_epochs_co = gr.Number(label="num_epochs")
            calico_button = gr.Button("DoCoodinate")
            loss_console_co = gr.Textbox(label="loss_console")
            plot_output_co = gr.Plot(label="実験結果のロス画像")
            plot_confirm_co = gr.Plot(label="実験結果の確認画像")
            
        with gr.TabItem("CaliOffset"):
            
            caliof_S_path = gr.Textbox(label="CaliOffset_Stopのフォルダのパス")
            caliof_T_path = gr.Textbox(label="CaliOffset_Trackのフォルダのパス")
            caliof_F_path = gr.Textbox(label="CaliOffset_Flickのフォルダのパス")
            caliof_outpath = gr.Textbox(label="CaliOffsetの出力フォルダのパス")
            batchsize_of = gr.Number(label="batchsize")
            num_epochs_of = gr.Number(label="num_epochs")
            M_of = gr.Number(label="inputsize")
            caliof_button = gr.Button("DoOffset")
            loss_console_of = gr.Textbox(label="loss_console")
            plot_output_of_mean = gr.Plot(label="実験結果のロス画像(平均)")
            plot_output_of_else = gr.Plot(label="実験結果のロス画像(それぞれ)")
            
    
    
    calico_button.click(calico.DoCaliCoodinate, 
                        inputs=[calico_path, calico_outoutpath, batchsize_co, num_epochs_co], 
                        outputs=[loss_console_co]).then(calico.Output_DoCaliCoodinste, None, [plot_output_co, plot_confirm_co])
            
    
    caliof_button.click(caliof.DoCaliOffset,
                        inputs=[caliof_S_path, caliof_T_path, caliof_F_path, caliof_outpath, batchsize_of, num_epochs_of, M_of],
                        outputs=[loss_console_of]).then(caliof.Output_DoCaliOffset, None, [plot_output_of_mean, plot_output_of_else])
                
demo.queue()   
# 起動
demo.launch()