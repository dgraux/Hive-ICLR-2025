HIVE Supplementary Material + MuSE Benchmark
============================================

> Associated with our ICLR'25 publication entitled "_From an LLM swarm
> to a PDDL-empowered HIVE: planning self-executed instructions in a
> multi-modal jungle_", we prepare an archive containing additional
> materials which may be useful.


File Hierarchy
--------------

This archive contains:

- `README.md` this file;
- `MuSE_Benchmark/` which provides the raw data for the MuSE benchmark
    we designed, composed of
   - `README.md`
   - `queries.json`
   - `data/audios/`
   - `data/images/`
- `C-KG_excerpt/` to visualise in a Web-browser the sub-graph of our
    Capability-KG corresponding to the experiments we present in the
    article so to tackle the MuSE benchmark.
   - `src/` containing various JavaScript data and library files
   - `capability_kg.html` the GUI to be opened
- `PDDL-domain-files/` corresponding to the 10 domains involved in the
    MuSE Benchmark
- `Screencast.mp4` which presents a complete run-through of a query
    for all the systems we tested in the submission, _i.e._,
    HuggingGPT, ControlLLM and **Hive**
- `Report.pdf` the associated report displayed in `Screencast.mp4`


Citation
--------

```
@misc{vyas2025hive,
     title={{From An LLM Swarm To A PDDL-Empowered HIVE: Planning
      Self-Executed Instructions In A Multi-Modal Jungle}},
     author={Kaustubh Vyas and Damien Graux and Yijun Yang and
      Sébastien Montella and Chenxin Diao and Wendi Zhou and Pavlos
      Vougiouklis and Ruofei Lai and Yang Ren and Keshuang Li and Jeff
      Z. Pan},
     year={2025},
     booktitle={The Thirteenth International Conference on Learning
      Representations, {ICLR} 2025, Singapore, April 24-28, 2025},
     publisher={OpenReview.net},
     url={https://arxiv.org/abs/2412.12839},
}
```
