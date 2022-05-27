`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/09/2021 03:05:30 PM
// Design Name: 
// Module Name: clk_div
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


module clk_div2(
    input wire clk_in,
    output wire clk_1
    );

wire din;
wire clkdiv;

dff dff_inst0 (
.clk(clk_in),
.rst(0),
.D(din),
.Q(clkdiv)
);

assign din = ~clkdiv;

assign clk_1 = clkdiv;

endmodule