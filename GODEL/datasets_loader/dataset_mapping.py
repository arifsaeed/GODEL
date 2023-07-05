def _dataset_mapping_function(examples, tokenizer, padding,args,column_mapping):

    contextes = examples[column_mapping['input']]
    responses = examples[column_mapping['label']]
    kbs = examples[column_mapping['knowledge']]

    inputs = []
    for context, kb in zip(contextes, kbs):
        if args.no_kb:
            inputs.append(context + ' => ')
        else:
            _input = context + ' <|knowledge|> ' + kb + ' => '
            inputs.append(_input)
    model_inputs = tokenizer(inputs, max_length=args.max_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(responses, max_length=args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["labels"]
    return model_inputs

def goal_directed_mapping_function(examples, **kwargs):
    tokenizer=kwargs.get("tokenizer")
    padding=kwargs.get("padding")
    args=kwargs.get("args")
    column_mapping={'input': 'Context','label':'Response','knowledge':'Knowledge'}
    return _dataset_mapping_function(examples, tokenizer, padding,args,column_mapping)

def chat_context_mapping_function(examples, **kwargs):
    tokenizer=kwargs.get("tokenizer")
    padding=kwargs.get("padding")
    args=kwargs.get("args")
    column_mapping = {'input': 'instruction','label':'response','knowledge':'context'}
    return _dataset_mapping_function(examples, tokenizer, padding,args,column_mapping)