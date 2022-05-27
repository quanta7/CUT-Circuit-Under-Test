`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/19/2021 06:07:09 PM
// Design Name: 
// Module Name: ro_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module ro_tb;

    reg clk_in;
    wire clk_0, clk_1, clk_14;
    
    design_1_wrapper dut(.clk_in(clk_in), .clk_0(clk_0), .clk_1(clk_1), .clk_14(clk_14));
    
    initial begin
    clk_in = 0;
    end
    
    always #2.5 clk_in <= ~clk_in;
    
endmodule
