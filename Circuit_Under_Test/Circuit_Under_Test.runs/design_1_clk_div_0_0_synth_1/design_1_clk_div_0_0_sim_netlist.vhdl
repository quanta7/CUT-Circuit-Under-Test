-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2020.1 (win64) Build 2902540 Wed May 27 19:54:49 MDT 2020
-- Date        : Wed Oct 20 12:54:29 2021
-- Host        : DESKTOP-EQ2MGIA running 64-bit major release  (build 9200)
-- Command     : write_vhdl -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
--               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ design_1_clk_div_0_0_sim_netlist.vhdl
-- Design      : design_1_clk_div_0_0
-- Purpose     : This VHDL netlist is a functional simulation representation of the design and should not be modified or
--               synthesized. This netlist cannot be used for SDF annotated simulation.
-- Device      : xc7z020clg400-1
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    q_reg_2 : in STD_LOGIC
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_2,
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    q_reg_2 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_2,
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1 is
  port (
    clkdiv : out STD_LOGIC_VECTOR ( 0 to 0 );
    q_reg_0 : in STD_LOGIC;
    q_reg_1 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_1,
      CE => '1',
      D => q_reg_0,
      Q => clkdiv(0),
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10 is
  port (
    clkdiv : out STD_LOGIC_VECTOR ( 0 to 0 );
    q_reg_0 : in STD_LOGIC;
    q_reg_1 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_1,
      CE => '1',
      D => q_reg_0,
      Q => clkdiv(0),
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    clkdiv : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => clkdiv(0),
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12 is
  port (
    clkdiv : out STD_LOGIC_VECTOR ( 0 to 0 );
    q_reg_0 : in STD_LOGIC;
    clk_in : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => clk_in,
      CE => '1',
      D => q_reg_0,
      Q => clkdiv(0),
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2 is
  port (
    clkdiv : out STD_LOGIC_VECTOR ( 0 to 0 );
    q_reg_0 : in STD_LOGIC;
    q_reg_1 : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_1(0),
      CE => '1',
      D => q_reg_0,
      Q => clkdiv(0),
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3 is
  port (
    q : out STD_LOGIC;
    D : in STD_LOGIC;
    clkdiv : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => clkdiv(0),
      CE => '1',
      D => D,
      Q => q,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    q : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q,
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    q_reg_2 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_2,
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6 is
  port (
    clkdiv : out STD_LOGIC_VECTOR ( 0 to 0 );
    q_reg_0 : in STD_LOGIC;
    q_reg_1 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_1,
      CE => '1',
      D => q_reg_0,
      Q => clkdiv(0),
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    clkdiv : in STD_LOGIC_VECTOR ( 0 to 0 )
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => clkdiv(0),
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    q_reg_2 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_2,
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9 is
  port (
    q_reg_0 : out STD_LOGIC;
    q_reg_1 : in STD_LOGIC;
    q_reg_2 : in STD_LOGIC
  );
  attribute ORIG_REF_NAME : string;
  attribute ORIG_REF_NAME of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9 : entity is "dff";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9 is
begin
q_reg: unisim.vcomponents.FDRE
     port map (
      C => q_reg_2,
      CE => '1',
      D => q_reg_1,
      Q => q_reg_0,
      R => '0'
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_div is
  port (
    clkdiv : out STD_LOGIC_VECTOR ( 4 downto 0 );
    clk_in : in STD_LOGIC
  );
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_div;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_div is
  signal D : STD_LOGIC;
  signal \^clkdiv\ : STD_LOGIC_VECTOR ( 4 downto 0 );
  signal \dff_gen_label[10].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[11].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[2].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[3].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[5].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[6].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[7].dff_inst_n_0\ : STD_LOGIC;
  signal \dff_gen_label[9].dff_inst_n_0\ : STD_LOGIC;
  signal q : STD_LOGIC;
  signal \q_i_1__10_n_0\ : STD_LOGIC;
  signal \q_i_1__11_n_0\ : STD_LOGIC;
  signal \q_i_1__12_n_0\ : STD_LOGIC;
  signal \q_i_1__1_n_0\ : STD_LOGIC;
  signal \q_i_1__2_n_0\ : STD_LOGIC;
  signal \q_i_1__3_n_0\ : STD_LOGIC;
  signal \q_i_1__4_n_0\ : STD_LOGIC;
  signal \q_i_1__5_n_0\ : STD_LOGIC;
  signal \q_i_1__6_n_0\ : STD_LOGIC;
  signal \q_i_1__7_n_0\ : STD_LOGIC;
  signal \q_i_1__8_n_0\ : STD_LOGIC;
  signal \q_i_1__9_n_0\ : STD_LOGIC;
  signal q_i_1_n_0 : STD_LOGIC;
begin
  clkdiv(4 downto 0) <= \^clkdiv\(4 downto 0);
\dff_gen_label[10].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff
     port map (
      q_reg_0 => \dff_gen_label[10].dff_inst_n_0\,
      q_reg_1 => \q_i_1__9_n_0\,
      q_reg_2 => \dff_gen_label[9].dff_inst_n_0\
    );
\dff_gen_label[11].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_0
     port map (
      q_reg_0 => \dff_gen_label[11].dff_inst_n_0\,
      q_reg_1 => \q_i_1__10_n_0\,
      q_reg_2 => \dff_gen_label[10].dff_inst_n_0\
    );
\dff_gen_label[12].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_1
     port map (
      clkdiv(0) => \^clkdiv\(3),
      q_reg_0 => \q_i_1__11_n_0\,
      q_reg_1 => \dff_gen_label[11].dff_inst_n_0\
    );
\dff_gen_label[13].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_2
     port map (
      clkdiv(0) => \^clkdiv\(4),
      q_reg_0 => \q_i_1__12_n_0\,
      q_reg_1(0) => \^clkdiv\(3)
    );
\dff_gen_label[1].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_3
     port map (
      D => D,
      clkdiv(0) => \^clkdiv\(0),
      q => q
    );
\dff_gen_label[2].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_4
     port map (
      q => q,
      q_reg_0 => \dff_gen_label[2].dff_inst_n_0\,
      q_reg_1 => \q_i_1__1_n_0\
    );
\dff_gen_label[3].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_5
     port map (
      q_reg_0 => \dff_gen_label[3].dff_inst_n_0\,
      q_reg_1 => \q_i_1__2_n_0\,
      q_reg_2 => \dff_gen_label[2].dff_inst_n_0\
    );
\dff_gen_label[4].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_6
     port map (
      clkdiv(0) => \^clkdiv\(1),
      q_reg_0 => \q_i_1__3_n_0\,
      q_reg_1 => \dff_gen_label[3].dff_inst_n_0\
    );
\dff_gen_label[5].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_7
     port map (
      clkdiv(0) => \^clkdiv\(1),
      q_reg_0 => \dff_gen_label[5].dff_inst_n_0\,
      q_reg_1 => \q_i_1__4_n_0\
    );
\dff_gen_label[6].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_8
     port map (
      q_reg_0 => \dff_gen_label[6].dff_inst_n_0\,
      q_reg_1 => \q_i_1__5_n_0\,
      q_reg_2 => \dff_gen_label[5].dff_inst_n_0\
    );
\dff_gen_label[7].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_9
     port map (
      q_reg_0 => \dff_gen_label[7].dff_inst_n_0\,
      q_reg_1 => \q_i_1__6_n_0\,
      q_reg_2 => \dff_gen_label[6].dff_inst_n_0\
    );
\dff_gen_label[8].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_10
     port map (
      clkdiv(0) => \^clkdiv\(2),
      q_reg_0 => \q_i_1__7_n_0\,
      q_reg_1 => \dff_gen_label[7].dff_inst_n_0\
    );
\dff_gen_label[9].dff_inst\: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_11
     port map (
      clkdiv(0) => \^clkdiv\(2),
      q_reg_0 => \dff_gen_label[9].dff_inst_n_0\,
      q_reg_1 => \q_i_1__8_n_0\
    );
dff_inst0: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_dff_12
     port map (
      clk_in => clk_in,
      clkdiv(0) => \^clkdiv\(0),
      q_reg_0 => q_i_1_n_0
    );
q_i_1: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \^clkdiv\(0),
      O => q_i_1_n_0
    );
\q_i_1__0\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => q,
      O => D
    );
\q_i_1__1\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[2].dff_inst_n_0\,
      O => \q_i_1__1_n_0\
    );
\q_i_1__10\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[11].dff_inst_n_0\,
      O => \q_i_1__10_n_0\
    );
\q_i_1__11\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \^clkdiv\(3),
      O => \q_i_1__11_n_0\
    );
\q_i_1__12\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \^clkdiv\(4),
      O => \q_i_1__12_n_0\
    );
\q_i_1__2\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[3].dff_inst_n_0\,
      O => \q_i_1__2_n_0\
    );
\q_i_1__3\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \^clkdiv\(1),
      O => \q_i_1__3_n_0\
    );
\q_i_1__4\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[5].dff_inst_n_0\,
      O => \q_i_1__4_n_0\
    );
\q_i_1__5\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[6].dff_inst_n_0\,
      O => \q_i_1__5_n_0\
    );
\q_i_1__6\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[7].dff_inst_n_0\,
      O => \q_i_1__6_n_0\
    );
\q_i_1__7\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \^clkdiv\(2),
      O => \q_i_1__7_n_0\
    );
\q_i_1__8\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[9].dff_inst_n_0\,
      O => \q_i_1__8_n_0\
    );
\q_i_1__9\: unisim.vcomponents.LUT1
    generic map(
      INIT => X"1"
    )
        port map (
      I0 => \dff_gen_label[10].dff_inst_n_0\,
      O => \q_i_1__9_n_0\
    );
end STRUCTURE;
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
  port (
    clk_in : in STD_LOGIC;
    clk_1 : out STD_LOGIC;
    clk_5 : out STD_LOGIC;
    clk_9 : out STD_LOGIC;
    clk_13 : out STD_LOGIC;
    clk_14 : out STD_LOGIC
  );
  attribute NotValidForBitStream : boolean;
  attribute NotValidForBitStream of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is true;
  attribute CHECK_LICENSE_TYPE : string;
  attribute CHECK_LICENSE_TYPE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "design_1_clk_div_0_0,clk_div,{}";
  attribute DowngradeIPIdentifiedWarnings : string;
  attribute DowngradeIPIdentifiedWarnings of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "yes";
  attribute IP_DEFINITION_SOURCE : string;
  attribute IP_DEFINITION_SOURCE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "module_ref";
  attribute X_CORE_INFO : string;
  attribute X_CORE_INFO of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix : entity is "clk_div,Vivado 2020.1";
end decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix;

architecture STRUCTURE of decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix is
begin
inst: entity work.decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_clk_div
     port map (
      clk_in => clk_in,
      clkdiv(4) => clk_14,
      clkdiv(3) => clk_13,
      clkdiv(2) => clk_9,
      clkdiv(1) => clk_5,
      clkdiv(0) => clk_1
    );
end STRUCTURE;
