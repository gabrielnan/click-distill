# Tzafon CUA Hackathon — Idea Pool

**Constraints**: 4hr research track. [Northstar-CUA-Fast](https://huggingface.co/Tzafon/Northstar-CUA-Fast) (4B, Apache-2.0, Qwen3-VL base). Nvidia compute credits. [Lightcone SDK](https://github.com/tzafon/lightcone) + [WayPoint browser fleet](https://github.com/tzafon/Tzafon-WayPoint). [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) (multimodal as of Feb 2026).

**Tzafon refs**: [GH org](https://github.com/tzafon) · [HF org](https://huggingface.co/Tzafon) · [VLM training blog](https://www.tzafon.ai/blog/training-vlm-for-cua) · [Northstar-CUA-Fast blog](https://www.tzafon.ai/blog/northstar-cua-fast)

**Key external refs**: [verifiers lib](https://github.com/PrimeIntellect-ai/verifiers) · [SimpleVLA-RL example](https://github.com/PRIME-RL/SimpleVLA-RL) · [UI-TARS-2 paper](https://arxiv.org/abs/2509.02544) · [OS-Atlas](https://arxiv.org/abs/2410.23218) · [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) · [Awesome-GUI-Agent](https://github.com/showlab/Awesome-GUI-Agent) · [CUA dataset aggregator](https://github.com/Khang-9966/Computer-Browser-Phone-Use-Agent-Datasets)

**Quick nav**: [Failure modes](#failure-mode-catalog) · [Themes](#theme-map) · [Round 1 ideas](#round-1--25-ideas-knowledge-derived-pre-research) · [Research findings](#research-wave-1--key-findings) · [Round 2 ideas](#round-2--25-more-ideas-research-derived) · [Ranking](#round-3--ranking) · [Combos](#recommended-hackathon-combos)

---

## Failure Mode Catalog

### Spatial/perception
- F1. Click off-by-N pixels (positional)
- F2. Coordinate frame confusion (zoom, DPI, viewport vs page)
- F3. Wrong-element-selected when adjacent UI is similar
- F4. Misreads numbers/dates (font + small target)
- F5. Loses track of cursor across scroll
- F6. Modal/popup occlusion misjudged (clicks underneath)

### Action/loop
- F7. Click-storm (repeated clicks same spot)
- F8. Stuck-in-loop (same action sequence)
- F9. Scroll exhaustion (infinite scroll page)
- F10. Premature terminate (stops before goal)
- F11. Late terminate (keeps acting after success)
- F12. Drag distance/velocity wrong
- F13. Right-click vs left-click confusion

### Planning/memory
- F14. Forgets subgoal mid-task
- F15. Distracted by side panel content
- F16. Can't recover after wrong-page navigation
- F17. No verification action emitted (assumes success)
- F18. Cross-tab state leakage
- F19. Repeats already-completed subtask

### Environmental
- F20. Auth wall / login prompt
- F21. CAPTCHA / bot detection
- F22. Cookie banner trap
- F23. Lazy-load not waited out
- F24. Animation in-progress (clicks moving target)
- F25. Right element exists in another iframe

### Model-internal
- F26. Coord-token loss treats 1px and 500px error equally
- F27. Vision encoder loses fine-grained position past N patches
- F28. Long-context drift (history pollutes attention)
- F29. Mode collapse (always emits same action type)
- F30. Refuses safe actions (overcautious)

---

## Theme Map

| Theme | Description | Hackathon fit |
|-------|-------------|---------------|
| T1. Positional encoding | Tzafon flagged as #1 bottleneck | High |
| T2. Recovery/robustness | Their stated future work | High |
| T3. Verification | Did the action succeed? | Medium |
| T4. Test-time search | Best-of-N, MCTS, verifier | High (no training) |
| T5. Synthetic envs | Their RL trained on only ~100 | High |
| T6. Multi-modal fusion | screenshot + DOM/AX tree | Medium |
| T7. Distillation | 4B → 1B for edge | Low (time) |
| T8. Probing/interpretability | Where does spatial info live? | High |
| T9. Calibration/uncertainty | When should agent ask for help? | Medium |
| T10. Curriculum design | Order RL envs for stability | Medium |
| T11. Cross-resolution generalization | Train 1080p, eval 4K | High |
| T12. Latency/speculative decode | Faster CUA inference | Medium |
| T13. Eval design | New benchmarks, failure tagging | High |
| T14. Reward shaping | Beyond bbox-hit | High |
| T15. Self-play / self-improve | Agent generates own envs | Low (time) |

---

## Round 1 — 25 Ideas (knowledge-derived, pre-research)

### Inference-only (lowest risk, fits 4hr easily)

<a id="i1"></a>**I1. Positional encoding scaling sweep** (T1, F1, F26)
[Tzafon showed](https://www.tzafon.ai/blog/training-vlm-for-cua) ×3 RoPE → 40%→80% on red-ball click. Sweep: ×1, ×2, ×3, ×4, ×6, learned-per-axis, NTK-style, YaRN-style. Eval on synthetic click bench + [ScreenSpot](https://github.com/njucckevin/SeeClick).

<a id="i2"></a>**I2. Test-time best-of-N with bbox vote** (T4, F1)
Sample N=8 actions, cluster click coords, take centroid of largest cluster. Cheap impl, likely +5-10pp on clicks.

<a id="i3"></a>**I3. Self-consistency for action type** (T4, F29)
Sample N completions, majority-vote action *type* before sampling coords. Reduces mode-collapse failures.

<a id="i4"></a>**I4. Verifier model second-pass** (T3, T4, F17)
After action, screenshot diff → small VLM judges "did this match intent?" If no, retry with action excluded. See [CUARewardBench](https://arxiv.org/html/2510.18596).

<a id="i5"></a>**I5. DOM-grounded coordinate snapping** (T6, F1, F2)
Post-process model's click → snap to nearest actual interactive element via accessibility tree. Hybrid neuro-symbolic.

<a id="i6"></a>**I6. Cursor-trail context injection** (T1, F5)
Render small marker at last-clicked position in next screenshot. Helps spatial continuity.

<a id="i7"></a>**I7. Multi-resolution ensemble** (T11, F27)
Run inference at 720p, 1080p, 1440p, vote. Different patch sizes capture different scales.

<a id="i8"></a>**I8. Zoom-on-uncertainty** (T9, F1)
If model's bbox confidence low, crop+upscale that region, re-query. Two-pass refinement.

<a id="i9"></a>**I9. Action delta gating** (T4, F7, F8)
If proposed action == last action AND screenshot didn't change > threshold, force exploration token.

### Eval/dataset (low risk, high OSS value)

<a id="i10"></a>**I10. ClickBench-Synthetic** (T13, F1, F4)
Procedurally generate 1K screenshots with known target widget locations. Release as HF dataset. Use to score [I1](#i1).

<a id="i11"></a>**I11. Failure taxonomy on OSWorld** (T13, F1-F30)
Run Northstar on [OSWorld](https://github.com/xlang-ai/OSWorld), auto-cluster failures via VLM judge. Tag traces. Release dataset.

<a id="i12"></a>**I12. Cross-DPI eval suite** (T11, T13)
Same task, same site, 1×/2×/3× device pixel ratio. Measure consistency.

<a id="i13"></a>**I13. Recovery eval** (T2, T13, F16)
Inject fault into trajectory at step k (wrong click), measure if model recovers. New metric: recovery@k.

<a id="i14"></a>**I14. Sub-step latency benchmark** (T12, T13)
Per-action latency profile on H100. Identifies bottleneck (vision encode? LM decode? action parse?).

### Training/RL (higher risk, needs Nvidia compute)

<a id="i15"></a>**I15. GRPO with recovery reward** (T2, T14)
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) multimodal + Northstar + reward = task_success - α·repeated_action_penalty + β·recover_after_fail_bonus.

<a id="i16"></a>**I16. Curriculum from synthetic to real** (T5, T10)
Generate synthetic envs, sort by step-length, train short→long. Compare vs random order.

<a id="i17"></a>**I17. Coord-loss replacement** (T1, T14, F26)
SFT with Gaussian-spread coordinate loss instead of cross-entropy. Distance-aware. See [GUI-G²](https://arxiv.org/html/2507.15846v2), [HyperClick](https://arxiv.org/html/2510.27266).

<a id="i18"></a>**I18. Multi-resolution fine-tune** (T11)
SFT on mixed 720/1080/1440 screenshots. Baseline = single-res.

<a id="i19"></a>**I19. Procedural web-env generator** (T5)
Playwright-driven pages with random forms/modals/menus. 1K env release on HF. Compare to [GUI-ReWalk](https://arxiv.org/html/2509.15738).

### Probing/analysis

<a id="i20"></a>**I20. Layer-wise spatial probe** (T8)
Linear probe at each vision-encoder layer for "where is widget X?" Shows where positional info dies.

<a id="i21"></a>**I21. Attention pattern on click** (T8)
Visualize which screenshot patches attend to coord tokens. Sanity check on F27.

<a id="i22"></a>**I22. Patch-token ablation** (T8)
Mask N% of patches at random positions, measure click-acc curve. How redundant is the patch grid?

### Architecture / inference

<a id="i23"></a>**I23. Speculative decoding for action heads** (T12)
Small Northstar drafts coord tokens, big Northstar verifies. Latency win. Background: [SpecVLM](https://arxiv.org/abs/2509.11815).

<a id="i24"></a>**I24. Tool-augmented planner** (T6, F14)
Wrap Northstar with explicit subgoal stack (text-only LM holds plan, Northstar executes step). Planner resets stack on goal change. See [Aguvis](https://arxiv.org/abs/2412.04454).

<a id="i25"></a>**I25. Refusal calibration** (T9, F30)
Detect when model is uncertain (entropy on coord head), emit `ask_user(question)` instead of click. Adapt [RefusalBench](https://arxiv.org/html/2510.10390).

---

## Research Wave 1 — Key Findings

1. **[prime-rl `verifiers`](https://github.com/PrimeIntellect-ai/verifiers) lib** = trajectory-level scalar reward via async Python rubric. **No CUA env shipped** = open slot in [Environments Hub](https://www.primeintellect.ai/blog/lab).
2. **[UI-TARS-2](https://arxiv.org/abs/2509.02544) (47.5 OSWorld)** beats Northstar via (a) multi-turn online RL at scale, (b) DPO on error corrections, (c) reflective trace bootstrapping. **Weights closed** ([1.5-7B is open](https://github.com/bytedance/ui-tars)) — replication is novel.
3. **Gaussian/distance-aware coord loss** has 4+ papers ([GUI-G²](https://arxiv.org/html/2507.15846v2), [HyperClick](https://arxiv.org/html/2510.27266), [AutoFocus](https://arxiv.org/abs/2605.02630v1), [ReGUIDE](https://arxiv.org/html/2505.15259)) but **none applied to Northstar / Qwen3-VL-4B-CUA end-to-end**. Standard is still cross-entropy on coord tokens.
4. **[OS-Atlas](https://arxiv.org/abs/2410.23218) grounding corpus = 13M elements** — largest open download. Cross-platform.
5. **[AITW](https://arxiv.org/pdf/2307.10088) = 715k Android episodes**, **[GUIEnv (GUICourse)](https://arxiv.org/html/2406.11317) = 10M UI pairs**, **[GUI-ReWalk](https://arxiv.org/html/2509.15738) / [GUIrilla](https://arxiv.org/html/2510.16051v1) / TreeCUA / DreamStruct** = synthetic-env tooling, all downloadable.
6. **[Aguvis](https://arxiv.org/abs/2412.04454)** = closest open planner-actor split. **OS-Atlas / ShowUI / [UGround](https://github.com/OSU-NLP-Group/UGround)** trend = vision-only (skip AX tree).
7. **[CUARewardBench](https://arxiv.org/html/2510.18596)**: step+trajectory reward-model eval; dominant errors = reasoning 35.8%, visual 30.2%. Verifier-model space is wide open.
8. **[RefusalBench](https://arxiv.org/html/2510.10390)**: 176 perturbation strategies — adaptable for CUA calibration eval.
9. **[WindowsAgentArena](https://microsoft.github.io/WindowsAgentArena/)**: 150 tasks, full run ~20 min on Azure parallel — feasible in 4hrs.
10. **Speculative decoding for VLMs**: [SpecVLM](https://arxiv.org/abs/2509.11815) / [ViSpec](https://arxiv.org/html/2509.15235v1) / [Spec-VLA](https://arxiv.org/abs/2507.22424) exist but **none target text+coord-token mixed output** for CUA.
11. **No latency / cross-resolution / recovery / refusal bench specifically for CUA** — multiple eval-design gaps. ([Scaling Agents](https://arxiv.org/html/2510.02250) covers trajectory-level only.)

---

## Round 2 — 25 More Ideas (research-derived)

### Tooling for the ecosystem (very high OSS value)

<a id="i26"></a>**I26. CUA env for prime-rl `verifiers` Hub** (T5, T14)
First Northstar-runnable env in the official [Environments Hub](https://www.primeintellect.ai/blog/lab). Wrap [Lightcone](https://github.com/tzafon/lightcone) harness, expose `load_environment()` + scalar reward via [verifiers](https://github.com/PrimeIntellect-ai/verifiers). Other researchers can fine-tune any VLM. Extremely high reuse potential.

<a id="i27"></a>**I27. Lightcone ↔ BrowserGym adapter** (T13)
[Lightcone](https://github.com/tzafon/lightcone) is new; [BrowserGym](https://github.com/ServiceNow/BrowserGym) is established (WebArena/WorkArena). Adapter lets Northstar run on existing benches without re-implementation. PR-able to lightcone.

<a id="i28"></a>**I28. OSWorld-to-prime-rl env wrapper** (T5)
[OSWorld](https://github.com/xlang-ai/OSWorld) is the canonical bench but VM-heavy. Package it as `verifiers.Environment` for direct RL fine-tuning loop.

### Coordinate loss / training (Gaussian gap)

<a id="i29"></a>**I29. HyperClick-style truncated Gaussian SFT on Northstar** (T1, T14, F26)
Apply [HyperClick](https://arxiv.org/html/2510.27266) loss to Qwen3-VL-4B Northstar. Variance scales with element bbox size. SFT on [OS-Atlas](https://arxiv.org/abs/2410.23218) 13M corpus subset. Targets [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981).

<a id="i30"></a>**I30. GUI-G² coverage reward ported to GRPO** (T2, T14)
[GUI-G²](https://arxiv.org/html/2507.15846v2) uses point + coverage Gaussian as RL reward. Prior work used PPO/AdamRL — port to [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) GRPO multimodal pipeline.

<a id="i31"></a>**I31. AutoFocus axial perplexity loss** (T1, F26)
[AutoFocus](https://arxiv.org/abs/2605.02630v1) anisotropic Gaussian (different x/y variance) for elongated UI elements (input fields wider than tall). Test if helps form-filling tasks.

<a id="i32"></a>**I32. Brier-calibrated click confidence** (T9)
[HyperClick](https://arxiv.org/html/2510.27266) uses Brier score for calibration. Output calibrated p(click_correct). Use for [I25](#i25) refusal trigger.

### Test-time / inference (no training)

<a id="i33"></a>**I33. Coord-clustering Best-of-N** (T4, F1) — *literature gap*
Sample N=16, K-means on coord pairs, return centroid of densest cluster. Drop variance threshold = "uncertain → ask user." Unified BoN + calibration. Most-related: [Scaling Agents for Computer Use](https://arxiv.org/html/2510.02250) (trajectory-level only).

<a id="i34"></a>**I34. Verifier-guided MCTS with screenshot diff reward** (T3, T4)
Lightweight MCTS over actions; reward = VLM-judged screenshot diff alignment to subgoal. Use Northstar as both policy + verifier.

<a id="i35"></a>**I35. Action-history compression via subgoal token** (F28)
Long history pollutes attention. Compress past N steps to "achieved: X, attempted: Y" via summary VLM call every K steps.

<a id="i36"></a>**I36. Speculative coord-token decoding** (T12) — *gap*
Tiny coord-only head drafts (x,y) tokens; full Northstar verifies. [SpecVLM](https://arxiv.org/abs/2509.11815) / [ViSpec](https://arxiv.org/html/2509.15235v1) / [Spec-VLA](https://arxiv.org/abs/2507.22424) didn't target mixed output. Latency win for CUA specifically.

<a id="i37"></a>**I37. Patch-attention crop pre-filter** (T1, F27)
Use vision encoder cross-attention on instruction text → top-K patches → crop to ROI → re-encode at higher resolution. Two-pass attention zoom.

<a id="i38"></a>**I38. Multi-aspect-ratio inference** (T11)
Native Qwen3-VL handles dynamic res. Run portrait crop + landscape crop, ensemble click predictions.

### Recovery / robustness (T2, biggest stated gap)

<a id="i39"></a>**I39. Counterfactual recovery dataset** (T2, T13, F16)
Take successful [OSWorld](https://github.com/xlang-ai/OSWorld) traces, perturb step k → wrong action. Continue with Northstar from perturbed state. Annotate whether recovery succeeded. Public dataset.

<a id="i40"></a>**I40. RL with self-correction reward** (T2, T14)
Reward shaping: r = success + α·(steps_after_first_failure_resolved). Trains model to detect+recover, not just avoid failure. Built on [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

<a id="i41"></a>**I41. Replay buffer of model's own failure traces** (T2, T15)
Fine-tune on (failed_state, recovery_action_demo). Demos generated by GPT-4o or human. Imitation-on-failures.

<a id="i42"></a>**I42. UI-TARS-2 reflective bootstrapping replication** (T2, T14)
[Their secret sauce](https://arxiv.org/abs/2509.02544): run agent → if fail, prompt for reflection → use reflection to generate corrected trace → SFT on corrected. Implement at small scale on Northstar.

### Eval design (T13, multiple gaps)

<a id="i43"></a>**I43. CUA-Latency-Bench** (T12, T13) — *gap*
Per-action latency × accuracy frontier across 3-4 OSS CUA models on H100. No published bench. Releasable. Models to include: Northstar, [UI-TARS-1.5-7B](https://github.com/bytedance/ui-tars), [OS-Atlas](https://arxiv.org/abs/2410.23218), [ShowUI](https://github.com/showlab/ShowUI).

<a id="i44"></a>**I44. CUA-Resolution-Bench** (T11, T13) — *gap*
Same task, 720p / 1080p / 1440p / 4K. Publish accuracy curve per model. Catches "trained on 1080p only" hidden weakness.

<a id="i45"></a>**I45. Recovery@k metric** (T2, T13)
Operationalize from [I39](#i39). Standardize the metric across OSS CUA models.

<a id="i46"></a>**I46. CUA-Refusal-Bench** (T9, T13) — *adapted*
Port [RefusalBench](https://arxiv.org/html/2510.10390) (176 perturbations) to CUA: ambiguous instructions, missing UI element, two valid targets. Measure refusal vs hallucinated action.

<a id="i47"></a>**I47. CUARewardBench-2 expansion** (T3, T13)
[CUARewardBench](https://arxiv.org/html/2510.18596) has limited corpus. Expand 5-10× via synthetic perturbation. Useful for verifier-model researchers.

### Vision encoder / probing

<a id="i48"></a>**I48. Patch-grid resolution probe** (T8, F27)
Vary number of patches Qwen3-VL sees (32×32, 64×64, 128×128). Plot click accuracy vs patch count. Identifies sweet spot.

<a id="i49"></a>**I49. Linear probe for "is this clickable?" per layer** (T8)
Classify if a patch belongs to interactive widget. Find layer where clickability emerges.

<a id="i50"></a>**I50. Encoder swap experiment** (T1, T8)
Replace Qwen3-VL vision tower with CLIP / SigLIP / DINO-v2 patches; freeze decoder; measure click accuracy. Identifies how much encoder matters.

### Multi-modal fusion (AX-tree gap)

<a id="i51"></a>**I51. AX-tree as auxiliary token stream** (T6) — *gap in vision-first OSS*
Inject AX tree text alongside screenshot. Train Northstar adapter to fuse. Counter to OSS trend ([UGround](https://github.com/OSU-NLP-Group/UGround) / [OS-Atlas](https://arxiv.org/abs/2410.23218) / [ShowUI](https://github.com/showlab/ShowUI) all skipped this); might unlock [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) big.

<a id="i52"></a>**I52. DOM-snap post-processor** (T6, F1, F2)
After model outputs (x,y), snap to nearest interactive DOM element bounding box. Pure Playwright post-process; no training. Should crush [ScreenSpot](https://github.com/njucckevin/SeeClick) subset.

<a id="i53"></a>**I53. OCR-augmented coord prediction** (T6)
Run Tesseract/PaddleOCR on screenshot, inject text+bbox triples as context. For text-target clicks specifically.

### Synthetic env / data

<a id="i54"></a>**I54. GUIrilla-style env in `verifiers`** (T5)
Wrap [GUIrilla](https://arxiv.org/html/2510.16051v1) automated UI exploration as RL env. Fast iteration without real apps.

<a id="i55"></a>**I55. Procedural form-filling env generator** (T5)
Playwright pages with random N-field forms (text, dropdown, date, file). Reward = filled correctly. Targets known weak area (F4, F12).

<a id="i56"></a>**I56. Adversarial UI perturbation generator** (T2, T5)
Modify real screenshots: shift elements N px, change colors, occlude with banners. Tests robustness. Cheap to generate.

### Distillation / efficiency

<a id="i57"></a>**I57. Northstar-Nano: distill 4B → 1B** (T7, T12)
Logit distill on [OS-Atlas](https://arxiv.org/abs/2410.23218) corpus. Edge / on-device deployment story. Risky in 4hrs unless we use existing distillation infra. Comparable: [Ferret-UI Lite (3B)](https://arxiv.org/abs/2509.26539).

<a id="i58"></a>**I58. LoRA-only adaptation per app** (T7)
One small LoRA per app (Chrome, Thunderbird, Sheets). Composable adapters. Tests if app-specific LoRAs beat monolithic fine-tune.

### Wild cards

<a id="i59"></a>**I59. Cross-model voting ensemble** (T4)
Northstar + [ShowUI](https://github.com/showlab/ShowUI) + [OS-Atlas](https://arxiv.org/abs/2410.23218) vote on click coords. Ensemble of small open CUAs. Compare to Northstar-alone + UI-TARS-2.

<a id="i60"></a>**I60. Self-play CUA-vs-CUA UI generation** (T15)
Generator agent builds adversarial UI; solver agent (Northstar) tries to complete. Co-evolution. Probably too ambitious for 4hr.

<a id="i61"></a>**I61. Northstar as autograder for human UI usability** (T13)
Inverse use: measure how often Northstar fails on a site → correlates with human usability. New metric: "agentic usability score." Publishable angle.

<a id="i62"></a>**I62. CUA-distilled keyboard-only mode** (T7, F30)
Force Northstar to use only keyboard navigation (Tab/Enter/arrows). Measures planning vs grounding decoupling. Accessibility implication.

<a id="i63"></a>**I63. Mid-trajectory replanning trigger** (T2, F14)
Detect "stuck" via screenshot-diff entropy. Trigger external planner LM to rewrite goal. Hybrid loop.

<a id="i64"></a>**I64. Gaze-prediction auxiliary head** (T8)
Predict where humans look on a screenshot (use [AITW](https://arxiv.org/pdf/2307.10088) or eye-tracking dataset proxy). Auxiliary loss during SFT. Spatial inductive bias.

<a id="i65"></a>**I65. Curriculum: ScreenSpot → ScreenSpot-Pro → OSWorld** (T10)
Stage SFT in difficulty order. Compare to random order. Standard curriculum hypothesis.

---

*End Round 2. 65 ideas total.*

---

## Round 3 — Ranking

**Scoring rubric** (1-5 each, 20 max):
- **Feas4h** = realistic in 4hrs?
- **Novel** = literature gap / extends Tzafon's stated future work?
- **OSS** = lasting contribution (PR, dataset, env, eval)?
- **Impact** = does it move Northstar closer to UI-TARS-2 / advance the field?

| ID | Idea | Feas4h | Novel | OSS | Impact | Total | Compute | Risk |
|----|------|--------|-------|-----|--------|-------|---------|------|
| [**I26**](#i26) | **CUA env for prime-rl `verifiers` Hub** | 5 | 5 | 5 | 4 | **19** | low | low |
| [**I33**](#i33) | **Coord-clustering Best-of-N** | 5 | 5 | 4 | 4 | **18** | low | low |
| [**I52**](#i52) | **DOM-snap post-processor** | 5 | 4 | 4 | 5 | **18** | none | very low |
| [**I29**](#i29) | **HyperClick Gaussian SFT on Northstar** | 4 | 5 | 4 | 5 | **18** | medium | low-med |
| [**I43**](#i43) | **CUA-Latency-Bench** | 5 | 4 | 5 | 3 | **17** | low | low |
| [**I44**](#i44) | **CUA-Resolution-Bench** | 5 | 4 | 5 | 3 | **17** | low | low |
| [**I1**](#i1) | **Positional encoding scaling sweep** | 5 | 4 | 4 | 4 | **17** | low | low |
| [**I39**](#i39) | **Counterfactual recovery dataset** | 4 | 5 | 5 | 3 | **17** | low | low |
| [**I42**](#i42) | UI-TARS-2 reflective bootstrap (mini) | 3 | 5 | 4 | 5 | 17 | medium | med |
| [**I46**](#i46) | CUA-Refusal-Bench (port RefusalBench) | 5 | 4 | 5 | 3 | 17 | low | low |
| [I8](#i8) | Zoom-on-uncertainty | 5 | 3 | 4 | 4 | 16 | none | low |
| [I40](#i40) | RL with self-correction reward | 3 | 5 | 4 | 4 | 16 | high | med-high |
| [I11](#i11) | Failure taxonomy on OSWorld | 4 | 3 | 5 | 4 | 16 | low | low |
| [I51](#i51) | AX-tree as auxiliary token stream | 3 | 5 | 4 | 4 | 16 | medium | med |
| [I36](#i36) | Speculative coord-token decoding | 3 | 5 | 4 | 4 | 16 | medium | med-high |
| [I30](#i30) | GUI-G² coverage reward → GRPO | 3 | 4 | 4 | 4 | 15 | high | med |
| [I20](#i20) | Layer-wise spatial probe | 4 | 4 | 3 | 4 | 15 | low | low |
| [I50](#i50) | Encoder swap (CLIP/SigLIP/DINO) | 3 | 4 | 3 | 5 | 15 | medium | med |
| [I53](#i53) | OCR-augmented coord prediction | 5 | 3 | 3 | 4 | 15 | none | low |
| [I32](#i32) | Brier-calibrated click confidence | 4 | 4 | 3 | 4 | 15 | low | low |

(Full 65-idea table truncated; remaining ideas tier-classified below.)

---

## Tier Summary

**S-tier (do this)** — top ROI, ship-shaped contributions:
- [**I26**](#i26) prime-rl env (ecosystem-defining, gap)
- [**I33**](#i33) coord-cluster BoN (literature gap, trivial impl)
- [**I52**](#i52) DOM-snap (huge ScreenSpot-Pro gain potential, near-zero risk)
- [**I29**](#i29) HyperClick Gaussian SFT (novel application of recent loss to Northstar)

**A-tier (strong second choices)**:
- [I43](#i43), [I44](#i44), [I46](#i46) (all eval benches — no compute, lasting value)
- [I1](#i1) (positional sweep — extends Tzafon's own ablation)
- [I39](#i39) (recovery dataset — high-value labels)
- [I42](#i42) (UI-TARS-2 replication — high impact if it lands)
- [I40](#i40) (recovery RL — best alignment with stated future work, but risky)

**B-tier (interesting, less hackathon-shaped)**:
[I2](#i2), [I3](#i3), [I4](#i4), [I5](#i5), [I7](#i7), [I8](#i8), [I11](#i11), [I20](#i20), [I32](#i32), [I34](#i34), [I36](#i36), [I49](#i49), [I50](#i50), [I51](#i51), [I53](#i53), [I54](#i54), [I55](#i55), [I63](#i63)

**C-tier (cool but ambitious for 4hrs)**:
[I15](#i15), [I17](#i17), [I18](#i18), [I22](#i22), [I30](#i30), [I41](#i41), [I47](#i47), [I48](#i48), [I57](#i57), [I58](#i58), [I59](#i59), [I64](#i64), [I65](#i65)

**D-tier (better as a follow-up paper)**:
[I6](#i6), [I9](#i9), [I12](#i12), [I13](#i13), [I14](#i14), [I16](#i16), [I19](#i19), [I21](#i21), [I23](#i23), [I24](#i24), [I25](#i25), [I27](#i27), [I28](#i28), [I31](#i31), [I35](#i35), [I37](#i37), [I38](#i38), [I45](#i45), [I56](#i56), [I60](#i60), [I61](#i61), [I62](#i62)

---

## Recommended Hackathon Combos

Pick one combo to ship as a coherent story.

### Combo α — "Inference-only robustness pack" (lowest risk)
[**I52**](#i52) + [**I33**](#i33) + [**I8**](#i8) + [**I1**](#i1) sweep on top.
Four post-hoc Northstar improvements stacked. Eval on ScreenSpot + ScreenSpot-Pro + small OSWorld subset. PR to lightcone. Story: "+X pp on Northstar at zero training cost."

### Combo β — "Ecosystem play" (highest OSS impact)
[**I26**](#i26) + [**I54**](#i54) (or [**I55**](#i55)).
Build the first CUA env in the prime-rl Hub, with one runnable synthetic env (GUIrilla-wrapped or form-filler). Anyone can now RL-train any VLM on CUA. Ship a 30-min RL curve as proof.

### Combo γ — "Loss function paper bait" (best for research narrative)
[**I29**](#i29) + [**I32**](#i32).
HyperClick Gaussian SFT on Northstar with Brier calibration head. Compare CE vs Gaussian on ScreenSpot-Pro. Releases: weights diff + LoRA + eval.

### Combo δ — "Eval suite drop" (no GPU needed)
[**I43**](#i43) + [**I44**](#i44) + [**I46**](#i46).
Three new CUA benches. Run baseline numbers on Northstar + 2-3 other open CUAs. Pure dataset/code release.

### Combo ε — "Recovery story" (best alignment with Tzafon's stated gaps, highest risk)
[**I39**](#i39) + [**I40**](#i40) (mini).
Build counterfactual recovery dataset (1hr), run small GRPO with self-correction reward (3hr). Even partial training curves are publishable.

---

## My recommendation

**Combo α** if you want a guaranteed shipped result in 4hrs.
**Combo β** if you care about lasting open-source impact above all else.
**Combo γ** if you want the cleanest research-paper narrative.

Want me to scope any combo into hour-by-hour milestones?
