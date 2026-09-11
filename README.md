<p align="center">
  <img src="https://github.com/user-attachments/assets/436635dc-0d6f-411a-9e55-73eb2bbc5a46" width="512" heigth="120">
</p>


<p align="center">
  A data compression framework developed as a project for the Compressione Dati (Data Compression) course, part of the Computer Science Master's Degree program at the University of Salerno
</p>


## Table of Contents
- [Overview](#Overview)


## Overview 
  This project builds on an existing architecture for semantic coding described by XXX et al., implementing that architecture and extending it with the addition of
  BCH codes with shorter parameters. Short BCH codes are used to encode and decode segments of phrases in parallel, cutting down on latency by processing multiple segments 
  simultaneously rather than sequentially. Any residual errors left uncorrected by the BCH codes are then resolved by an AI model, which leverages the semantic context of the 
  content to identify and fix these errors as a semantic task, rather than relying on additional redundancy.
