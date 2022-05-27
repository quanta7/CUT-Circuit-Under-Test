// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2020.1 (win64) Build 2902540 Wed May 27 19:54:49 MDT 2020
// Date        : Wed Oct 20 12:54:29 2021
// Host        : DESKTOP-EQ2MGIA running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ design_1_clk_div_0_0_sim_netlist.v
// Design      : design_1_clk_div_0_0
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z020clg400-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_div
   (clkdiv,
    clk_in);
  output [4:0]clkdiv;
  input clk_in;

  wire D;
  wire clk_in;
  wire [4:0]clkdiv;
  wire \dff_gen_label[10].dff_inst_n_0 ;
  wire \dff_gen_label[11].dff_inst_n_0 ;
  wire \dff_gen_label[2].dff_inst_n_0 ;
  wire \dff_gen_label[3].dff_inst_n_0 ;
  wire \dff_gen_label[5].dff_inst_n_0 ;
  wire \dff_gen_label[6].dff_inst_n_0 ;
  wire \dff_gen_label[7].dff_inst_n_0 ;
  wire \dff_gen_label[9].dff_inst_n_0 ;
  wire q;
  wire q_i_1__10_n_0;
  wire q_i_1__11_n_0;
  wire q_i_1__12_n_0;
  wire q_i_1__1_n_0;
  wire q_i_1__2_n_0;
  wire q_i_1__3_n_0;
  wire q_i_1__4_n_0;
  wire q_i_1__5_n_0;
  wire q_i_1__6_n_0;
  wire q_i_1__7_n_0;
  wire q_i_1__8_n_0;
  wire q_i_1__9_n_0;
  wire q_i_1_n_0;

  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff \dff_gen_label[10].dff_inst 
       (.q_reg_0(\dff_gen_label[10].dff_inst_n_0 ),
        .q_reg_1(q_i_1__9_n_0),
        .q_reg_2(\dff_gen_label[9].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0 \dff_gen_label[11].dff_inst 
       (.q_reg_0(\dff_gen_label[11].dff_inst_n_0 ),
        .q_reg_1(q_i_1__10_n_0),
        .q_reg_2(\dff_gen_label[10].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1 \dff_gen_label[12].dff_inst 
       (.clkdiv(clkdiv[3]),
        .q_reg_0(q_i_1__11_n_0),
        .q_reg_1(\dff_gen_label[11].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2 \dff_gen_label[13].dff_inst 
       (.clkdiv(clkdiv[4]),
        .q_reg_0(q_i_1__12_n_0),
        .q_reg_1(clkdiv[3]));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3 \dff_gen_label[1].dff_inst 
       (.D(D),
        .clkdiv(clkdiv[0]),
        .q(q));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4 \dff_gen_label[2].dff_inst 
       (.q(q),
        .q_reg_0(\dff_gen_label[2].dff_inst_n_0 ),
        .q_reg_1(q_i_1__1_n_0));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5 \dff_gen_label[3].dff_inst 
       (.q_reg_0(\dff_gen_label[3].dff_inst_n_0 ),
        .q_reg_1(q_i_1__2_n_0),
        .q_reg_2(\dff_gen_label[2].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6 \dff_gen_label[4].dff_inst 
       (.clkdiv(clkdiv[1]),
        .q_reg_0(q_i_1__3_n_0),
        .q_reg_1(\dff_gen_label[3].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7 \dff_gen_label[5].dff_inst 
       (.clkdiv(clkdiv[1]),
        .q_reg_0(\dff_gen_label[5].dff_inst_n_0 ),
        .q_reg_1(q_i_1__4_n_0));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8 \dff_gen_label[6].dff_inst 
       (.q_reg_0(\dff_gen_label[6].dff_inst_n_0 ),
        .q_reg_1(q_i_1__5_n_0),
        .q_reg_2(\dff_gen_label[5].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9 \dff_gen_label[7].dff_inst 
       (.q_reg_0(\dff_gen_label[7].dff_inst_n_0 ),
        .q_reg_1(q_i_1__6_n_0),
        .q_reg_2(\dff_gen_label[6].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10 \dff_gen_label[8].dff_inst 
       (.clkdiv(clkdiv[2]),
        .q_reg_0(q_i_1__7_n_0),
        .q_reg_1(\dff_gen_label[7].dff_inst_n_0 ));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11 \dff_gen_label[9].dff_inst 
       (.clkdiv(clkdiv[2]),
        .q_reg_0(\dff_gen_label[9].dff_inst_n_0 ),
        .q_reg_1(q_i_1__8_n_0));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12 dff_inst0
       (.clk_in(clk_in),
        .clkdiv(clkdiv[0]),
        .q_reg_0(q_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1
       (.I0(clkdiv[0]),
        .O(q_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__0
       (.I0(q),
        .O(D));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__1
       (.I0(\dff_gen_label[2].dff_inst_n_0 ),
        .O(q_i_1__1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__10
       (.I0(\dff_gen_label[11].dff_inst_n_0 ),
        .O(q_i_1__10_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__11
       (.I0(clkdiv[3]),
        .O(q_i_1__11_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__12
       (.I0(clkdiv[4]),
        .O(q_i_1__12_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__2
       (.I0(\dff_gen_label[3].dff_inst_n_0 ),
        .O(q_i_1__2_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__3
       (.I0(clkdiv[1]),
        .O(q_i_1__3_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__4
       (.I0(\dff_gen_label[5].dff_inst_n_0 ),
        .O(q_i_1__4_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__5
       (.I0(\dff_gen_label[6].dff_inst_n_0 ),
        .O(q_i_1__5_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__6
       (.I0(\dff_gen_label[7].dff_inst_n_0 ),
        .O(q_i_1__6_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__7
       (.I0(clkdiv[2]),
        .O(q_i_1__7_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__8
       (.I0(\dff_gen_label[9].dff_inst_n_0 ),
        .O(q_i_1__8_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    q_i_1__9
       (.I0(\dff_gen_label[10].dff_inst_n_0 ),
        .O(q_i_1__9_n_0));
endmodule

(* CHECK_LICENSE_TYPE = "design_1_clk_div_0_0,clk_div,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "module_ref" *) 
(* X_CORE_INFO = "clk_div,Vivado 2020.1" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (clk_in,
    clk_1,
    clk_5,
    clk_9,
    clk_13,
    clk_14);
  input clk_in;
  output clk_1;
  output clk_5;
  output clk_9;
  output clk_13;
  output clk_14;

  wire clk_1;
  wire clk_13;
  wire clk_14;
  wire clk_5;
  wire clk_9;
  wire clk_in;

  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_div inst
       (.clk_in(clk_in),
        .clkdiv({clk_14,clk_13,clk_9,clk_5,clk_1}));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff
   (q_reg_0,
    q_reg_1,
    q_reg_2);
  output q_reg_0;
  input q_reg_1;
  input q_reg_2;

  wire q_reg_0;
  wire q_reg_1;
  wire q_reg_2;

  FDRE q_reg
       (.C(q_reg_2),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0
   (q_reg_0,
    q_reg_1,
    q_reg_2);
  output q_reg_0;
  input q_reg_1;
  input q_reg_2;

  wire q_reg_0;
  wire q_reg_1;
  wire q_reg_2;

  FDRE q_reg
       (.C(q_reg_2),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1
   (clkdiv,
    q_reg_0,
    q_reg_1);
  output [0:0]clkdiv;
  input q_reg_0;
  input q_reg_1;

  wire [0:0]clkdiv;
  wire q_reg_0;
  wire q_reg_1;

  FDRE q_reg
       (.C(q_reg_1),
        .CE(1'b1),
        .D(q_reg_0),
        .Q(clkdiv),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10
   (clkdiv,
    q_reg_0,
    q_reg_1);
  output [0:0]clkdiv;
  input q_reg_0;
  input q_reg_1;

  wire [0:0]clkdiv;
  wire q_reg_0;
  wire q_reg_1;

  FDRE q_reg
       (.C(q_reg_1),
        .CE(1'b1),
        .D(q_reg_0),
        .Q(clkdiv),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11
   (q_reg_0,
    q_reg_1,
    clkdiv);
  output q_reg_0;
  input q_reg_1;
  input [0:0]clkdiv;

  wire [0:0]clkdiv;
  wire q_reg_0;
  wire q_reg_1;

  FDRE q_reg
       (.C(clkdiv),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12
   (clkdiv,
    q_reg_0,
    clk_in);
  output [0:0]clkdiv;
  input q_reg_0;
  input clk_in;

  wire clk_in;
  wire [0:0]clkdiv;
  wire q_reg_0;

  FDRE q_reg
       (.C(clk_in),
        .CE(1'b1),
        .D(q_reg_0),
        .Q(clkdiv),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2
   (clkdiv,
    q_reg_0,
    q_reg_1);
  output [0:0]clkdiv;
  input q_reg_0;
  input [0:0]q_reg_1;

  wire [0:0]clkdiv;
  wire q_reg_0;
  wire [0:0]q_reg_1;

  FDRE q_reg
       (.C(q_reg_1),
        .CE(1'b1),
        .D(q_reg_0),
        .Q(clkdiv),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3
   (q,
    D,
    clkdiv);
  output q;
  input D;
  input [0:0]clkdiv;

  wire D;
  wire [0:0]clkdiv;
  wire q;

  FDRE q_reg
       (.C(clkdiv),
        .CE(1'b1),
        .D(D),
        .Q(q),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4
   (q_reg_0,
    q_reg_1,
    q);
  output q_reg_0;
  input q_reg_1;
  input q;

  wire q;
  wire q_reg_0;
  wire q_reg_1;

  FDRE q_reg
       (.C(q),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5
   (q_reg_0,
    q_reg_1,
    q_reg_2);
  output q_reg_0;
  input q_reg_1;
  input q_reg_2;

  wire q_reg_0;
  wire q_reg_1;
  wire q_reg_2;

  FDRE q_reg
       (.C(q_reg_2),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6
   (clkdiv,
    q_reg_0,
    q_reg_1);
  output [0:0]clkdiv;
  input q_reg_0;
  input q_reg_1;

  wire [0:0]clkdiv;
  wire q_reg_0;
  wire q_reg_1;

  FDRE q_reg
       (.C(q_reg_1),
        .CE(1'b1),
        .D(q_reg_0),
        .Q(clkdiv),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7
   (q_reg_0,
    q_reg_1,
    clkdiv);
  output q_reg_0;
  input q_reg_1;
  input [0:0]clkdiv;

  wire [0:0]clkdiv;
  wire q_reg_0;
  wire q_reg_1;

  FDRE q_reg
       (.C(clkdiv),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8
   (q_reg_0,
    q_reg_1,
    q_reg_2);
  output q_reg_0;
  input q_reg_1;
  input q_reg_2;

  wire q_reg_0;
  wire q_reg_1;
  wire q_reg_2;

  FDRE q_reg
       (.C(q_reg_2),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "dff" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9
   (q_reg_0,
    q_reg_1,
    q_reg_2);
  output q_reg_0;
  input q_reg_1;
  input q_reg_2;

  wire q_reg_0;
  wire q_reg_1;
  wire q_reg_2;

  FDRE q_reg
       (.C(q_reg_2),
        .CE(1'b1),
        .D(q_reg_1),
        .Q(q_reg_0),
        .R(1'b0));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;
    parameter GRES_WIDTH = 10000;
    parameter GRES_START = 10000;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    wire GRESTORE;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;
    reg GRESTORE_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;
    assign (strong1, weak0) GRESTORE = GRESTORE_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

    initial begin 
	GRESTORE_int = 1'b0;
	#(GRES_START);
	GRESTORE_int = 1'b1;
	#(GRES_WIDTH);
	GRESTORE_int = 1'b0;
    end

endmodule
`endif
