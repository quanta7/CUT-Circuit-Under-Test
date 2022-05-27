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


module clk_div(
    input wire clk_in,
    output wire clk_1,
	output wire clk_5,
	output wire clk_9,
	output wire clk_13,
	output wire clk_14
    );

wire [13:0] din;
wire [13:0] clkdiv;

dff dff_inst0 (
.clk(clk_in),
.rst(0),
.D(din[0]),
.Q(clkdiv[0])
);

genvar i;
generate
for (i = 1; i < 14; i=i+1) 
begin : dff_gen_label
	dff dff_inst (
		.clk(clkdiv[i-1]),
		.rst(0),
		.D(din[i]),
		.Q(clkdiv[i])
	);
	end
endgenerate

assign din = ~clkdiv;

assign clk_1 = clkdiv[0];
assign clk_5 = clkdiv[4];
assign clk_9 = clkdiv[8];
assign clk_13 = clkdiv[12];
assign clk_14 = clkdiv[13];

endmodule