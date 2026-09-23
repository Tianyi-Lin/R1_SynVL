def consensus(judgements, names):
    if set(judgements) != set(names) or any(type(value) is not bool for value in judgements.values()):
        raise ValueError("Consensus requires one boolean judgement for every configured model")
    score = sum(judgements.values())
    return {
        "consensus_score": score,
        "num_judges": len(names),
        "qualified": score > 0,
        "adversarial": 0 < score < len(names),
        "judgements": judgements,
    }


def judge_candidate(backend, candidate, names, config):
    answers = {}
    for name in names:
        model = next(model for model in config["models"] if model["name"] == name)
        instruction = (
            "Solve the question using only its text. No image is available to you. "
            "If essential visual information is missing, answer INSUFFICIENT_INFORMATION rather than inventing it. "
            if model["visual_mode"] == "text_only" else
            "Solve the question using the attached visual transcription. You do not directly see the image. "
            "If essential evidence is marked uncertain or missing, answer INSUFFICIENT_INFORMATION rather than inventing it. "
            if model["visual_mode"] == "caption" else "Solve the question using the image. "
        )
        answers[name] = backend.ask(
            name, instruction + "Return {answer: string, explanation: string}. "
            "For MCQ return only the option letter as answer; otherwise return the minimal final answer.",
            {"question": candidate["new_question_text"]}, [candidate["generated_image_path"]], ("answer", "explanation"),
        )
        answers[name] = {**answers[name], "input_mode": model["visual_mode"]}
        if model["visual_mode"] == "caption":
            answers[name]["caption_model"] = config["caption_model"]
    judgements = {}
    for name, answer in answers.items():
        if answer["answer"].strip() == candidate["new_question_answer"].strip():
            judgements[name] = True
        else:
            def validate_match(value):
                if type(value.get("equivalent")) is not bool:
                    raise ValueError("equivalent must be boolean")
                return value
            match = backend.ask(
                config["equivalence_model"],
                "Compare final answers for semantic equivalence in the question's context. "
                "Do not solve again or repair the reference. Ambiguous or incomplete predictions are incorrect. "
                "Return {equivalent: boolean, reason: string}.",
                {"question": candidate["new_question_text"], "reference": candidate["new_question_answer"], "prediction": answer["answer"]},
                fields=("reason",), validator=validate_match,
            )
            judgements[name] = match["equivalent"]
    return {"model_answers": answers, **consensus(judgements, names)}
