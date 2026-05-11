# AI, ML and Generative AI in the Data Lakehouse

Official companion repository for **AI, ML and Generative AI in the Data Lakehouse** by Bennie Haelen, published by O'Reilly in 2026.

This repository contains the notebooks, code examples, and supporting assets referenced throughout the book. The material is organized by chapter so you can follow along as you read, or jump directly to the topic you are working on.

## About the book

The book is a practitioner's guide to building production AI, machine learning, and generative AI workloads on the data lakehouse. It covers the architectural patterns, governance practices, and engineering workflows that make the lakehouse a credible foundation for modern AI, with hands-on examples on the Databricks Lakehouse Platform.

## Repository structure

The code is organized by chapter. Each chapter folder contains the notebooks and supporting files for the examples in that chapter.

| Folder | Topic |
|---|---|
| `chapter04` | End-to-End ML with MLflow |
| `chapter05` | Feature Engineering in the Unity Catalog |
| `chapter06` | ML at Scale |
| `chapter07` | GenAI in the Lakehouse: Foundations and Architecture|
| `chapter09` | AI Agents in the Lakehouse |
| `chapter10` | The Model Context Protocol|
| `chapter11` | Agent-to-Agent Communication and the DSPy Framework |
| `common` | Shared utilities and helper code used across chapters |
| `Various notebooks for later consideration` | Reference and exploratory notebooks not directly tied to a chapter |

## Getting started

The notebooks are designed to run on the Databricks Lakehouse Platform. To use them:

1. Clone this repository into your Databricks workspace as a Git folder, or download and import the notebooks directly.
2. Attach the notebooks to a cluster with the runtime version specified at the top of each notebook.
3. Follow along with the corresponding chapter in the book.

Some chapters require additional setup (datasets, model registry entries, Unity Catalog objects). Setup instructions are included in the chapter folder where they apply.

## Requirements

- A Databricks workspace (Community Edition works for most examples; some chapters use features available only in the Premium or Enterprise tiers)
- Databricks Runtime version as noted per chapter
- Access to Unity Catalog for chapters covering governance, lineage, and the model registry

## Errata and feedback

If you find an issue with the code or have a suggestion, please open an issue in this repository. For errata in the book text itself, please use the O'Reilly errata system on the book's catalog page.

## License

The code in this repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## About the author

Bennie Haelen is a Principal Data and AI Architect at Insight, an O'Reilly author, and co-author of *Delta Lake: Up and Running*. He works with large enterprises on data lakehouse architecture, governed AI platforms, and production machine learning systems.
