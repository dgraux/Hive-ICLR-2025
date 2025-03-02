# MuSE: Multi-modal Sub-task Execution

MuSE is a comprehensive benchmark designed to evaluate solutions for multifaceted multi-modal AI tasks.

## Details

MuSE includes a total of 100 queries, categorized by the number of tasks involved in each query as summarized in the table below.

| Task Type   | No of Queries |
|-------------|---------------|
| Single Task | 25            |
| Two Tasks   | 60            |
| Three Tasks | 15            |

### Data Point Structure

Each data point in the benchmark consists of the following fields:

- **query**: The user query that serves as the input.
- **idx**: Index number of the query.
- **input_types**: The types of input present in the query.
- **output_types**: The type of output expected.
- **domains**: Related domains for the query.
- **tasks**: Specific tasks from the related domains that need to be addressed to find a solution for the query.

### Domains Covered

The queries are related to the following 10 domains:

- Audio
- Image Generation
- Image to Text
- Image to Image
- Machine Translation
- Question Answering
- Summarization
- Text Generation
- Text Classification
- Token Classification

## Suggested Metrics

Evaluation of AI systems using the MuSE benchmark can be based on three key metrics:

1. **Task Selection (TS)**: This metric assesses whether the system accurately identifies the required tasks from the user's query. Correct task selection is crucial for laying the foundation for successful execution and directly affects the relevance of the final output.

2. **Flow of Thought (FoT)**: This metric evaluates the logical sequence and integration of the selected tasks. It ensures that, particularly for multi-task queries involving two or three tasks, the system processes tasks in an order that leads to the desired outcome.

3. **Final Output (O)**: This metric assesses the correctness of the system's final response to the user's query. It includes evaluating the accuracy of answers, the relevance of generated content, and the overall satisfaction of the user's intent.