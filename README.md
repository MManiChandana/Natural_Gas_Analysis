# Energy Price Forecasting, Storage Contract Valuation & Credit Risk Modeling

## Project Overview

This project is a **finance and risk analytics portfolio project** that combines **commodity price forecasting**, **storage contract valuation**, and **credit risk prediction** using **Python, data analysis, and machine learning**.

The project is divided into **two major modules**, each solving a practical problem commonly seen in **energy markets, banking, and financial risk management**.

---

# Natural Gas Price Forecasting

---

## Objective

The goal of this module is to analyze historical **monthly natural gas price data** and estimate the gas price for **any historical date** as well as **forecast prices one year into the future**.

The dataset contains monthly natural gas purchase prices from:

- **31st October 2020**
- to
- **30th September 2024**

## What this module does

- Loads and analyzes monthly natural gas price data
- Visualizes historical trends
- Detects seasonal movement in prices
- Forecasts natural gas prices for future dates
- Provides a function to estimate gas price for any selected date

## Business Relevance

Natural gas prices often fluctuate due to:

- seasonal demand
- weather conditions
- supply disruptions
- energy consumption trends
- storage and inventory effects

Forecasting these prices is useful for **energy trading, planning, and storage strategy decisions**.

---
# Module 1B: storage Contract Valuation

---

## Objectives

Using the forecasted natural gas prices, this module estimates the value of a **natural gas storage contract**.

A storage contract allows a trader or company to:

- **inject gas into storage when prices are low**
- **withdraw and sell gas later when prices are high**

## Inputs considered in pricing

The contract pricing model takes into account:

- Injection dates
- Withdrawal dates
- Purchase prices
- Selling prices
- Injection / withdrawal rates
- Maximum storage volume
- Storage costs

## Core Pricing Logic

The value of the storage contract is calculated by estimating the **profit or loss** generated from each storage cycle:

```text
Cycle Profit = (Sell Price - Buy Price) × Volume - Storage Cost
