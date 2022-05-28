
# coding: utf-8

# In[10]:


from pynq import Overlay
from time import sleep
import numpy as np
from os import path
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
import email
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pickle as pkl


# In[11]:

class TestChip(Overlay):
    """TestChip class is the main driver
    class for interacting with our FPGA bitstream
    Coded by: Uncle Arash
    Version: 0.0
    """

    def __init__(self, ol_path, **kwargs):
        super().__init__(ol_path)
        self.heater_base_address = 0x00000000
        self.RO_base_address = 0x00000000
        self.BTI_base_write_address = 0x00000000  # select read sensor
        self.BTI_base_read_address = 0x00000004  # select read sensor
        self.HCI_base_freq_address = 0x00000000
        self.HCI_base_duty_address = 0x00000004
        self.HCI_base_write_address = 0x00000008
        self.HCI_base_read_address = 0x0000000C
        self.BTI_base_write_address = 0x00000000
        self.BTI_base_read_address = 0x00000004
        self.temp_sensor_address = 0x200
        self.vccint_sensor_address = 0x204
        self.vccaux_sensor_address = 0x208
        self.Vp_Vn_sensor_address = 0x20C
        self.vrefP_sensor_address = 0x210
        self.vrefN_sensor_address = 0x214
        self.vbram_sensor_address = 0x218
        self.pssvccint_sensor_address = 0x234
        self.pssvccaux_sensor_address = 0x238
        self.pssvccmem_sensor_address = 0x23C
        self.counter_address_increament = 0x04
        self.num_oscillators = 31
        self.num_BTI = 31
        self.intensity_dict = {i: int(sum(
            [2**j for j in range(i)])
        ) for i in range(1, 33)
        }
        self.intensity_dict[0] = 0
        self.sensor_dict = {i: 2**i for i in range(32)}
        for key, value in kwargs.items():
            if "heater_base_address" in key:
                self.heater_base_address = value
            if "counter_base_address" in key:
                self.counter_base_address = value
            if "num_oscillators" in key:
                self.num_oscillators = value
            if "counter_address_increament" in key:
                self.counter_address_increament = value
            if "temp_sensor_address" in key:
                self.temp_sensor_address = value
        self._a = 4.07548611   # 5
        self._b = 0.50103761   # 50
        self.temp_ctrl_sensitivity = 2
        self.temp_ctrl_intensity = 0

    def en_feedback(self, enable):
        if enable:
            self.RO2_0.write(0x00000000, 1048575)
            self.RO2_1.write(0x00000000, 1048575)
            self.RO2_2.write(0x00000000, 1048575)
            self.RO2_3.write(0x00000000, 1048575)
            self.RO2_4.write(0x00000000, 1048575)
            self.RO2_5.write(0x00000000, 1048575)
            #self.BTI0.write(0x00000000, 127)
            #self.BTI1.write(0x00000000, 127)
        else:
            self.RO2_0.write(0x00000000, 0)
            self.RO2_1.write(0x00000000, 0)
            self.RO2_2.write(0x00000000, 0)
            self.RO2_3.write(0x00000000, 0)
            self.RO2_4.write(0x00000000, 0)
            self.RO2_5.write(0x00000000, 0)
            #self.BTI0.write(0x00000000, 0)
            #self.BTI1.write(0x00000000, 0)

    def XADC_temp(self):
        return ((self.temp_sensor.read(self.temp_sensor_address
                                       ) >> 4) * 503.975/4096 - 273.15)

    def XADC_voltage(self, voltage_name='vccint'):
        if voltage_name == 'vccaux':
            measurement_out = self.temp_sensor.read(self.vccaux_sensor_address)
        elif voltage_name == 'vbram':
            measurement_out = self.temp_sensor.read(self.vbram_sensor_address)
        elif voltage_name == 'pssvccint':
            measurement_out = self.temp_sensor.read(self.pssvccint_sensor_address)
        elif voltage_name == 'pssvccaux':
            measurement_out = self.temp_sensor.read(self.pssvccaux_sensor_address)
        elif voltage_name == 'pssvccmem':
            measurement_out = self.temp_sensor.read(self.pssvccmem_sensor_address)
        else:
            measurement_out = self.temp_sensor.read(self.vccint_sensor_address)
        return (measurement_out >> 4) * 3/4096

    def freq2temp(self, del_f):
        """The values of a and b are for 5 stage ROs in ZYNQ 7000"""
        return del_f * self._a + self._b

    def read_RO(self, RO_list):
        """Reads the frequency of selected ROs
        Parameters:
        RO_list (list of int): list of ROs whose values we read

        Returns: freq_list (nparray)
        """
        len_ro = len(RO_list)
        freq_list = np.zeros((len_ro))
        for i in range(len_ro):
            assert RO_list[i] <= self.num_oscillators
            freq_list[i] = self.RO0.read(
                self.RO_base_address +
                RO_list[i] * self.counter_address_increament
            )/1000
        return freq_list

    def read_multi_RO(self, RO_dict):
        """Reads the frequency of selected ROs
        Parameters:
        RO_dict : {keys=RO_ip_name, values=[RO list per IP]}

        Returns: freq_dict {keys=RO_ip_name, values=[frequencies (nparray)]}
        """
        freq_dict = {}
        for item in RO_dict.keys():
            RO_list = RO_dict[item]
            len_ro = len(RO_list)
            freq_list = np.zeros((len_ro))
            RO = getattr(self, item)
            for i in range(len_ro):
                freq_list[i] = RO.read(
                    self.RO_base_address +
                    RO_list[i] * self.counter_address_increament
                )/1000
            freq_dict[item] = freq_list
        return freq_dict

    def read_BTI(self, BTI_list):
        """Reads the frequency of selected BTI sensort
        Parameters:
        BTI_list (list of int): list of BTI sensors whose values we read

        Returns: freq_list (nparray)
        """
        len_ro = len(BTI_list)
        freq_list = np.zeros((len_ro))
        for i in range(len_ro):
            assert BTI_list[i] <= self.num_BTI
            #   Putting the BTI sensors into the counting mode
            self.BTI0.write(self.BTI_base_write_address, self.sensor_dict[BTI_list[i]])
            freq_list[i] = self.BTI0.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            #   Putting the BTI sensors back into the aging mode
            self.BTI0.write(self.BTI_base_write_address, 0)
        return freq_list

    def read_multi_BTI(self, BTI_dict):
        """Reads the frequency of selected BTI sensort
        Parameters:
        BTI_dict: {keys=BTI_ip_name, values=[BTI list per IP]}

        Returns: freq_dict {keys=BTI_ip_name, values=[frequencies (nparray)]}
        """
        freq_dict = {}
        for item in BTI_dict.keys():
            BTI_list = BTI_dict[item]
            len_ro = len(BTI_list)
            freq_list = np.zeros((len_ro))
            BTI = getattr(self, item)
            for i in range(len_ro):
                #   Putting the BTI sensors into the counting mode
                BTI.write(self.BTI_base_write_address, self.sensor_dict[BTI_list[i]])
                freq_list[i] = BTI.read(
                    self.BTI_base_read_address +
                    BTI_list[i] * self.counter_address_increament
                )/1000
                #   Putting the BTI sensors back into the aging mode
                BTI.write(self.BTI_base_write_address, 0)
        return freq_dict

    def read_BTI_RO(self, BTI_list):
        """Reads the frequency of selected BTI sensort
        Parameters:
        BTI_list (list of int): list of BTI sensors whose values we read

        Returns: freq_list (nparray)
        """
        len_ro = len(BTI_list)
        freq_list = np.zeros((len_ro, 2))
        for i in range(len_ro):
            assert BTI_list[i] <= self.num_BTI
            #   Putting the BTI sensors into the counting mode
            ol.LowFreq_100Hz.write(0x0, 0)
            freq_list[i, 0] = self.BTI_RO1.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 1] = self.BTI_RO2.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            #   Putting the BTI sensors back into the aging mode
            ol.LowFreq_100Hz.write(0x0, 1)
        return freq_list

    def read_HCI(self, BTI_list):
        """Reads the frequency of selected BTI sensort
        Parameters:
        BTI_list (list of int): list of BTI sensors whose values we read

        Returns: freq_list (nparray)
        """
        len_ro = len(BTI_list)
        freq_list = np.zeros((len_ro, 7))
        for i in range(len_ro):
            assert BTI_list[i] <= self.num_BTI
            #   Putting the BTI sensors into the counting mode
#             self.HCI_5.write(self.BTI_base_write_address, self.sensor_dict[BTI_list[i]])
            self.HCI_0.write(0x0, 0)
            self.HCI_1.write(0x0, 0)
            self.HCI_2.write(0x0, 0)
            self.HCI_3.write(0x0, 0)
            self.HCI_4.write(0x0, 0)
            self.HCI_5.write(0x0, 0)
            self.HCI_6.write(0x0, 0)
#             ol.LowFreq_100Hz.write(0x0,0)
            freq_list[i, 0] = self.HCI_0.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 1] = self.HCI_1.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 2] = self.HCI_2.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 3] = self.HCI_3.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 4] = self.HCI_4.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 5] = self.HCI_5.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            freq_list[i, 6] = self.HCI_6.read(
                self.BTI_base_read_address +
                BTI_list[i] * self.counter_address_increament
            )/1000
            #   Putting the BTI sensors back into the aging mode
            self.HCI_0.write(0x0, 1)
            self.HCI_1.write(0x0, 1)
            self.HCI_2.write(0x0, 1)
            self.HCI_3.write(0x0, 1)
            self.HCI_4.write(0x0, 1)
            self.HCI_5.write(0x0, 1)
            self.HCI_6.write(0x0, 1)
#             ol.LowFreq_100Hz.write(0x0,0)
        return freq_list

    def arash_HCI_set_pwm(self, HCI_list, input_clock_frequency=100000000, output_clock_frequency=100, duty_cycle=50):
        """Sets the frequency and duty cycle of the HCI sensor
        Parameters:
        HCI_list : [HCI_ip_name]

        Returns: None
        """
        for item in HCI_list:
            HCI = getattr(self, item)
            HCI.write(self.HCI_base_freq_address, int(
                input_clock_frequency / output_clock_frequency))
            sleep(0.01)
            HCI.write(self.HCI_base_duty_address, int((duty_cycle / 100)
                      * input_clock_frequency / output_clock_frequency))
            sleep(0.01)

    def arash_read_HCI(self, HCI_dict):
        """Reads the frequency of selected HCI sensort
        Parameters:
        HCI_dict: {keys=HCI_ip_name, values=[HCI list per IP]}

        Returns: freq_dict {keys=HCI_ip_name, values=[frequencies (nparray)]}
        """
        freq_dict = {}
        for item in HCI_dict.keys():
            HCI_list = HCI_dict[item]
            len_ro = len(HCI_list)
            assert len_ro < 29
            freq_list = np.zeros((len_ro))
            HCI = getattr(self, item)
            for i in range(len_ro):
                #   Putting the BTI sensors into the counting mode
                HCI.write(self.HCI_base_write_address, self.sensor_dict[HCI_list[i]])
                sleep(0.01)
                freq_list[i] = HCI.read(
                    self.HCI_base_read_address +
                    HCI_list[i] * self.counter_address_increament
                )/1000
                #   Putting the BTI sensors back into the aging mode
                HCI.write(self.HCI_base_write_address, 0)
            freq_dict[item] = freq_list
        return freq_dict

    def top_region_heat_on(self, intensity):
        """Turns the top region heat on
        top region heat, heats up the whole top region of the chip

        Parameters:
        intensity (int): intensity of the heat
        (the final temperature depends on ventilation
        and/or isolation of the chip)
        intensity has to be between 0 to 64

        Returns: None
        """

        if intensity > 64:
            intensity = 64
        elif intensity < 0:
            intensity = 0

        if intensity < 33:
            lsb_heater = self.intensity_dict[intensity]
            msb_heater = 0x00000000
        else:
            lsb_heater = 0xFFFFFFFF
            msb_heater = self.intensity_dict[intensity-32]

        self.heater.write(self.heater_base_address, lsb_heater)
        self.heater.write(self.heater_base_address+0x04, msb_heater)

    def top_region_heat_off(self):
        """Turns the top region heat off
        top region heat, heats up the whole top region of the chip

        Returns: None
        """

        self.heater.write(self.heater_base_address, 0x00000000)
        self.heater.write(self.heater_base_address+0x04, 0x00000000)

    def fix_temperature(self, desired_temperature):
        """simple control scheme to fix the temperature to a desired value

        Returns: None
        """

        if self.XADC_temp() > (desired_temperature +
                               self.temp_ctrl_sensitivity):
            self.temp_ctrl_intensity -= 1
            self.top_region_heat_on(self.temp_ctrl_intensity)
        elif self.XADC_temp() < (desired_temperature -
                                 self.temp_ctrl_sensitivity):
            self.temp_ctrl_intensity += 1
            self.top_region_heat_on(self.temp_ctrl_intensity)
        if self.temp_ctrl_intensity > 34:
            self.temp_ctrl_intensity = 34
        elif self.temp_ctrl_intensity < 0:
            self.temp_ctrl_intensity = 0

    def stabilize_temperature(self, temperature, duration=int, step=2):
        """simple control scheme to gradually increase or decrease the
        temperature to a desired value

        Returns: None
        """
        print(
            f'Stabilizing the temperature at {temperature}' +
            f' Current temperature is: {self.XADC_temp()}\n'
        )
        init_temp = self.XADC_temp()
        delta = (temperature - init_temp) / (duration//step)
        for i in range(duration//step + 1):
            self.fix_temperature(init_temp + i * delta)
            sleep(step)
        print(f'Stabilized temperature is: {self.XADC_temp()}\n')

    def __str__(self):
        return (f"Number of ROs: {self.num_oscillators}; "
                "Current temperature: {self.XADC_temp()}; "
                "Number of BTIs: {self.num_BTI}")

    def sendemail(self, subject, filename, receiver_email):
        #subject = "Ring Oscillator Data"
        body = "This is an email with .pkl attachment sent from Pynq board"
        sender_email = "###@gmail.com" #Mention your email address
#         receiver_email = "ringoscillatorsensor@gmail.com"
        password = "###" #Mention your email account's password

        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject

        # Add body to email
        message.attach(MIMEText(body, "plain"))

        # filename = "Instname_1_1.pkl"  # In same directory as script

        # Open PDF file in binary mode
        with open(filename, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {filename}",
        )

        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()

        # Log in to server using secure context and send email
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, text)
        except:
            pass


# In[12]:


def record(ol, total_duration, every, num_oscillators, desired_temp):
    """
    parameters
    ----------
    ol              : FPGA bitstream file location 
    total_duration  : total duration in seconds
    every           : sampling step size in seconds
    num_oscillators : number of CUTs per instantiation of CUT Module
    desired_temp    : desired stress temperature
    """
    
    data = []   # [RO0, ..., RO{num_oscillators},Temperature, En_Freq, Duty_Cycle]
    RO_dict = dict(RO2_0=list(range(num_oscillators)), RO2_1=list(range(num_oscillators)), RO2_2=list(range(num_oscillators)), RO2_3=list(range(num_oscillators)),
                   RO2_4=list(range(num_oscillators)), RO2_5=list(range(num_oscillators)))
    times = []
    init_time = datetime.now()
    now_time = init_time
    while(now_time < (init_time + total_duration)):
        ol.fix_temperature(desired_temp)
        now_time = datetime.now()

        # Temperature
        temp = ol.XADC_temp()

        # Voltage
        vccint = ol.XADC_voltage('vccint')
        vccaux = ol.XADC_voltage('vccaux')
        vbram = ol.XADC_voltage('vbram')
        pssvccint = ol.XADC_voltage('pssvccint')
        pssvccaux = ol.XADC_voltage('pssvccaux')
        pssvccmem = ol.XADC_voltage('pssvccmem')

        # Stack all the measurements
        output_dict = ol.read_multi_RO(RO_dict)
        current_read = np.hstack((output_dict['RO2_0'], output_dict['RO2_1']))
        current_read = np.hstack((current_read, output_dict['RO2_2']))
        current_read = np.hstack((current_read, output_dict['RO2_3']))
        current_read = np.hstack((current_read, output_dict['RO2_4']))
        current_read = np.hstack((current_read, output_dict['RO2_5']))
        current_read = np.hstack((current_read, np.array([temp])))

        current_read = np.hstack((current_read, np.array([vccint])))
        current_read = np.hstack((current_read, np.array([vccaux])))
        current_read = np.hstack((current_read, np.array([vbram])))
        current_read = np.hstack((current_read, np.array([pssvccint])))
        current_read = np.hstack((current_read, np.array([pssvccaux])))
        current_read = np.hstack((current_read, np.array([pssvccmem])))

        now_pacific_live = datetime.now(timezone('US/Pacific'))
        times.append(now_pacific_live)

        data.append(current_read)

        while(datetime.now() < now_time + every):
            if(every > timedelta(seconds=1)):
                sleep(1)
                ol.fix_temperature(desired_temp)
            pass

    data = np.vstack(data)

    output = pd.DataFrame(data, columns=([f'RO2_0{i}' for i in range(num_oscillators)] +
                                         [f'RO2_1{i}' for i in range(num_oscillators)] +
                                         [f'RO2_2{i}' for i in range(num_oscillators)] +
                                         [f'RO2_3{i}' for i in range(num_oscillators)] +
                                         [f'RO2_4{i}' for i in range(num_oscillators)] +
                                         [f'RO2_5{i}' for i in range(
                                             num_oscillators)] + ['Temperature']
                                         + ['vccint'] + ['vccaux'] + ['vbram'] + ['pssvccint']
                                         + ['pssvccaux'] + ['pssvccmem']))

    output['Timestamp'] = pd.DataFrame(dict(Timestamp=times))

    return output


# In[9]:

now_pacific = datetime.now(timezone('US/Pacific'))

pre_every = timedelta(seconds=12)
pre_total_duration = timedelta(minutes=5)
stress_total_duration = timedelta(hours=1)
stress_every = timedelta(minutes=1)
measure_duration = timedelta(minutes=2)
measure_every = timedelta(seconds=12)

##################################################################################################
#Pre Stess
ol = TestChip(f'/home/xilinx/pynq/overlays/###/###.bit') #Mention the location of the FPGA bitstream
ol.en_feedback(1)
for k in range(50):
    sleep(2)
    ol.fix_temperature(125)

temp_data = np.arange(130,141,2);
for j in range(temp_data.shape[0]):
    
    for k in range(20):
        sleep(1)
        ol.fix_temperature(temp_data[j])
    
    output = record(ol, pre_total_duration, pre_every, 20, temp_data[j])
    output.to_pickle(f'./###{j}.pkl') #Name your pickle file
    
    #send email to Parvez
    filename = f'###{j}.pkl' #Mention your pickle filename
    subject = f'###{j}' #Give a unique subject to your email
    receiver_email = '###@gmail.com' #Mention the receiver's email address
    ol.sendemail(subject,filename,receiver_email)

##################################################################################################
#Stress
for k in range(20):
    sleep(5)
    ol.fix_temperature(135)

for i in range(1001):
    ol.en_feedback(0)
    output = record(ol, stress_total_duration, stress_every, 20, 135)
    output.to_pickle(f'./###{i}.pkl')  #Name your pickle file
    
    ol.en_feedback(1)
    output = record(ol, measure_duration, measure_every, 20, 135)
    output.to_pickle(f'./###{i}.pkl')  #Name your pickle file

    # send email to Parvez
    filename = f'###{i}.pkl' #Mention your pickle filename
    subject = f'[###{i}' #Give a unique subject to your email
    receiver_email = '###@gmail.com' #Mention the receiver's email address
    ol.sendemail(subject, filename, receiver_email)

    filename = f'###{i}.pkl' #Mention your pickle filename
    subject = f'[###{i}' #Give a unique subject to your email
    receiver_email = '###5@gmail.com' #Mention the receiver's email address
    ol.sendemail(subject, filename, receiver_email)
##################################################################################################
"""#Post Stess
ol.en_feedback(1)
for k in range(50):
    sleep(2)
    ol.fix_temperature(115)

temp_data = np.arange(120,131,2);
for j in range(temp_data.shape[0]):
    
    for k in range(20):
        sleep(1)
        ol.fix_temperature(temp_data[j])
    
    output = record(ol, pre_total_duration, pre_every, 20, temp_data[j])
    output.to_pickle(f'./Parvez_RO{j}_poststress_temp_vs_freq.pkl') #Parvez
    
    #send email to Parvez
    filename = f'Parvez_RO{j}_poststress_temp_vs_freq.pkl'
    subject = f'[Experiment_8]Parvez Post_Stress data for Temp_vs_freq - {j}'
    receiver_email = 'projectms555@gmail.com'
    ol.sendemail(subject,filename,receiver_email)
"""##################################################################################################
# Replace the overlay
ol = TestChip(f'/home/xilinx/pynq/overlays/###/###.bit') #Mention the address of the bitstream you want to load the end of the experiment
# Replace the overlay
