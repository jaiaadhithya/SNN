<div align="center">

# Neuromorphic Memristor Robot  
### FPGA-Based SNN with 3×3 Sensor Mesh and Trainable Memristor Crossbar



</div>

---

## Project Overview

This project implements a **physical neuromorphic robot** using:

- A 3×3 LDR sensor mesh  
- A memristor crossbar  
- An FPGA-implemented spiking neural network  
- A geometric weight assignment logic  
- A button-based training system  

The robot learns to associate spatial light patterns with motor control decisions.  
Learning happens directly on memristor hardware, not in software.

Reference for weight assignment logic:  
`sandbox:/mnt/data/Weight Assignment Logic - PDF (1).pdf`

---

## System Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {
  "primaryColor": "#0f172a",
  "secondaryColor": "#1f2937",
  "tertiaryColor": "#2563eb",
  "primaryTextColor": "#ffffff",
  "lineColor": "#60a5fa"
}}}%%

flowchart TB

  subgraph Sensors[3x3 LDR Sensor Grid]
    L1[LDR 0,0]:::sensor
    L2[LDR 0,1]:::sensor
    L3[LDR 0,2]:::sensor
    L4[LDR 1,0]:::sensor
    L5[LDR 1,1]:::sensor
    L6[LDR 1,2]:::sensor
    L7[LDR 2,0]:::sensor
    L8[LDR 2,1]:::sensor
    L9[LDR 2,2]:::sensor
  end

  subgraph Analog[Analog Frontend]
    ADC[ADC]
    TIA[Current Sense and TIA]
    DAC[DAC and Pulse Driver]
  end

  subgraph Crossbar[Memristor Layer]
    Xbar[Memristor Crossbar 9x2]:::memristor
  end

  subgraph FPGA[FPGA System]
    SNN[Spiking Neural Network]
    Train[Training Controller]
  end

  subgraph Actuation[Motor Control]
    Driver[Motor Driver]
    LM[Left Motor]
    RM[Right Motor]
  end

  Buttons[Training Buttons]:::button

  Sensors --> ADC
  ADC --> FPGA
  FPGA --> DAC
  DAC --> Xbar
  Xbar --> TIA --> ADC
  FPGA --> Driver
  Driver --> LM
  Driver --> RM
  Buttons --> Train
  Train --> FPGA

  classDef sensor fill:#2563eb,stroke:#0f172a,color:white;
  classDef memristor fill:#9333ea,stroke:#0f172a,color:white;
  classDef motor fill:#dc2626,stroke:#0f172a,color:white;
  classDef button fill:#16a34a,stroke:#0f172a,color:white;
