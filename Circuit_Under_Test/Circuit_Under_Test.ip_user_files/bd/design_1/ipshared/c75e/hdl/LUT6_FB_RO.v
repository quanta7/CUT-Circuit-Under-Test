`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/04/2021 01:01:27 PM
// Design Name: 
// Module Name: LUT6_FB_RO
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


module LUT6_FB_RO(
    (*DONT_TOUCH= "true"*) input En,
    (*DONT_TOUCH= "true"*) input r_out,
    (*DONT_TOUCH= "true"*) input f_i,
    (*DONT_TOUCH= "true"*) output r_in,
    (*DONT_TOUCH= "true"*) output freq_count
    );
// LUT6_2: 6-input, 2 output Look-Up Table
// 7 Series
// Xilinx HDL Libraries Guide, version 14.7
//(* LOCK_PINS="I0:A1, I1:A2, I2:A3, I3:A4, I4:A5, I5:A6" *)
LUT6_2 #(
.INIT(64'hCACACACACACACACA)
) LUT6_2_inst (
.O6(r_in), // 1-bit LUT6 output
.O5(freq_count), // 1-bit lower LUT5 output
.I0(f_i), // 1-bit LUT input
.I1(r_out), // 1-bit LUT input
.I2(En), // 1-bit LUT input
.I3(1'b0), // 1-bit LUT input
.I4(1'b0), // 1-bit LUT input
.I5(1'b0) // 1-bit LUT input (fast MUX select only available to O6 output)
);
endmodule