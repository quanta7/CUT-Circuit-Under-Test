`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/09/2021 03:06:10 PM
// Design Name: 
// Module Name: dff
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


module dff(
    input D,
    input clk,
    input rst,
    output Q
    );
	
reg q;
assign Q = q;
always @ (posedge(clk), posedge(rst))
begin
    if (rst == 1) begin
        q <= 1'b0;
    end
	else begin
		q <= D;
	end
end

endmodule