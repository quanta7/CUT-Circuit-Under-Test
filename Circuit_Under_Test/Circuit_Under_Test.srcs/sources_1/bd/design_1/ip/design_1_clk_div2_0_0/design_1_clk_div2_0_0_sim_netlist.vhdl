-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2020.1 (win64) Build 2902540 Wed May 27 19:54:49 MDT 2020
-- Date        : Wed Oct 20 12:55:24 2021
-- Host        : DESKTOP-EQ2MGIA running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim
--               c:/Tutorials/Vivado/parvez_ro_exp6/parvez_ro_exp6.srcs/sources_1/bd/design_1/ip/design_1_clk_div2_0_0/design_1_clk_div2_0_0_sim_netlist.vhdl
-- Design      : design_1_clk_div2_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7z020clg400-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity design_1_clk_div2_0_0_dff is
  port (
    clk_1 : out STD_LOGIC;
    clk_in : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of design_1_clk_div2_0_0_dff : entity is "dff";
end design_1_clk_div2_0_0_dff;

architecture STRUCTURE of design_1_clk_div2_0_0_dff is
  signal \^clk_1\ : STD_LOGIC;
  signal din : STD_LOGIC;
begin
  clk_1 <= \^clk_1\;
q_i_1: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \^clk_1\,
      O => din
    );
q_reg: unisim.vcomponents.FDRE
     port map (
      C => clk_in,
      CE => '1',
      D => din,
      Q => \^clk_1\,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity design_1_clk_div2_0_0_clk_div2 is
  port (
    clk_1 : out STD_LOGIC;
    clk_in : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of design_1_clk_div2_0_0_clk_div2 : entity is "clk_div2";
end design_1_clk_div2_0_0_clk_div2;

architecture STRUCTURE of design_1_clk_div2_0_0_clk_div2 is
begin
dff_inst0: entity work.design_1_clk_div2_0_0_dff
     port map (
      clk_1 => clk_1,
      clk_in => clk_in
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity design_1_clk_div2_0_0 is
  port (
    clk_in : in STD_LOGIC;
    clk_1 : out STD_LOGIC
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of design_1_clk_div2_0_0 : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of design_1_clk_div2_0_0 : entity is "design_1_clk_div2_0_0,clk_div2,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of design_1_clk_div2_0_0 : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of design_1_clk_div2_0_0 : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of design_1_clk_div2_0_0 : entity is "clk_div2,Vivado 2020.1";
end design_1_clk_div2_0_0;

architecture STRUCTURE of design_1_clk_div2_0_0 is
begin
inst: entity work.design_1_clk_div2_0_0_clk_div2
     port map (
      clk_1 => clk_1,
      clk_in => clk_in
    );
end STRUCTURE;
