import marimo

__generated_with = "0.14.7"
app = marimo.App(html_head_file="head.html")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Context/Memory Engineering Workshop for AI Founders

    Workshop Goal: Understand when different context strategies fail or succeed,
    and how to validate accuracy with your own documents.

    /// admonition | Source Code
    GitHub => https://github.com/Enverge-Labs/Memory-Context-and-RAG-Comparison-Tool
    ///

    /// attention | Important!
    Use company-specific information that is NOT in the LLM's training data.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Overview & Table of Contents

    This notebook walks through two stages:

    1. **Single-PDF mini-pipeline** — ask questions over one document.  
    2. **SDN database pipeline** — scale to a multi-folder corpus with retrieval choices and trade-offs.

    > **Learning outcomes**
    > - Explain why we chunk long docs and how overlap preserves meaning.
    > - Compare embedding backends (local vs OpenAI) and reason about latency/cost/quality.
    > - Use folder scoping (e.g., Policies) to balance trust/compliance vs completeness.
    > - Describe FAISS as the scaling backbone for fast retrieval.


    /// details | Table of Contents

    **Part A — Single PDF (mini-RAG)**

    - A1. Select & load a PDF
    - A2. Strategy 1 — Brute Force Memory stuffing
    - A3. Strategy 2 — Keyword RAG
    - A4. Strategy 3 — Semantic RAG
    - A5. Strategy 4 — Hybrid + AI Double-Check
    - A6. Tricky questions (failure-mode probes)

    **Part B — SDN Database (scaling up)**

    - B1. Setup / installs
    - B2. Colab dataset setup
    - B3. Crawl SDN folders
    - B4. Chunk documents
    - B5. Embeddings + cache (local vs OpenAI)
    - B6. Status summary
    - B7. Build FAISS index
    - B8. Semantic search helper
    - B9. Mini-RAG over SDN
    - B10. Practice prompts
    - B11. Experiments dashboard
    ///
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Part A: Single-PDF section explainers""")
    return


@app.cell(hide_code=True)
def _(mo):
    openrouter_api_key = mo.ui.text(value="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", full_width=True)

    mo.callout(mo.md(f"""
        ### Setup

        Input your API key for OpenRouter.ai below: {openrouter_api_key}
    """))
    return (openrouter_api_key,)


@app.cell(hide_code=True)
def _(mo, openrouter_api_key):
    mo.output.replace(mo.status.spinner(title="Loading AI memory system (embedding model)..."))

    from openai import OpenAI
    import time
    import pandas as pd
    import json
    import pymupdf as fitz # for PDF processing
    import os
    import numpy as np
    from sentence_transformers import SentenceTransformer

    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=openrouter_api_key.value,
    )

    # Initialize embedding model for semantic search
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    mo.output.replace(mo.callout(
        mo.vstack([
            mo.md(":white_check_mark: AI memory system ready!"),
            mo.md(":white_check_mark: Setup complete! Ready to test different AI memory strategies."),
            mo.md("""
                /// details | SELECTED AI MODELS - Memory & Performance Analysis:

                - MoonshotAI Kimi K2: HEAVY MODEL (1T params, 32B active)
                - TNG DeepSeek R1T Chimera: SPEED OPTIMIZED
                - DeepSeek R1: REASONING FOCUSED
                - GLM 4.5 Air: LIGHTWEIGHT (Compact MoE)
                - DeepSeek V3.1: FLAGSHIP BALANCED

                Watch how each model trades off SPEED vs ACCURACY vs MEMORY USAGE!
                ///
                """)
        ]),
        kind="success"
    ))

    def print_performance_metrics(response, time_elapsed, method_name, chunks_info="", is_correct=None, approach="", model="", test_question="", mo_output_extension=[]):
        """Helper function to display results for YOUR analysis"""

        response_time = mo.stat(
            value=f"{time_elapsed:.2f}s", 
            label="Response Time", 
            #caption="12k from last month", 
            #direction="increase"
        )

        memory_used = mo.stat(
            value=f"{response.usage.total_tokens}", 
            label="Memory Used (tokens)", 
            #caption="12k from last month", 
            #direction="increase"
        )

        memory_cost = mo.stat(
            value=f"${(response.usage.total_tokens * 0.001):.4f}", 
            label="Memory Cost", 
            #caption="12k from last month", 
            #direction="increase"
        )

        print_vstack = [
            mo.md(f"### {method_name} Results"),
            mo.hstack([response_time, memory_used, memory_cost], justify="center", gap="2rem"),
            mo.md(f"**Approach:** {approach}"),
            mo.md(f"**Model:** {model}"),
            mo.md(f"**Question:** {test_question}")
        ]

        if chunks_info:
            print_vstack.append(mo.md(f"**Information Retrieved:** {chunks_info}"))

        if is_correct is not None:
            status = ":white_check_mark: FOUND CORRECT INFO" if is_correct else ":x: MISSING/WRONG INFO"
            print_vstack.append(mo.md(f"**Accuracy:** {status}"))

        # Check for reasoning tokens (some models like R1 show thinking process)
        answer_text = response.choices[0].message.content
        if "<think>" in answer_text or "thinking:" in answer_text.lower():
            print_vstack.append(f"🧠 Reasoning overhead detected (extra memory usage)")

        print_vstack.append(mo.md(f"**Answer:** {response.choices[0].message.content}"))

        colour = "success" if is_correct else "danger"

        print_vstack.extend(mo_output_extension)

        mo_output = mo.callout(mo.vstack(print_vstack),kind=colour)

        return {
            "method": method_name,
            "latency": time_elapsed,
            "total_tokens": response.usage.total_tokens,
            "answer": response.choices[0].message.content,
            "is_correct": is_correct,
            "mo_output": mo_output
        }

    def check_answer_accuracy(question, answer, validation_questions):
        """Check if answer contains key information"""
        if question not in validation_questions:
            return None

        answer_lower = answer.lower()
        key_info = validation_questions[question]["key_info"]

        matches = sum(1 for info in key_info if info.lower() in answer_lower)
        is_correct = matches >= len(key_info) * 0.7  # 70% of key info must be present

        return is_correct
    return (
        check_answer_accuracy,
        client,
        embedding_model,
        fitz,
        np,
        print_performance_metrics,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Single PDF
    **What this does:** Points to one PDF (e.g., a policy or spec) that we’ll query.  
    **Why:** A minimal, “closed world” makes retrieval quality easy to see before we scale up.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, pdf_file, use_sample_data_switch):
    def upload_block(sample_data=True):
        if not sample_data:
            return mo.md(f"""
                Upload your company data in PDF format

                {pdf_file}
                """)
        else:
            return ""

    mo.callout(
        mo.vstack([
            mo.md("### Company Data + PDF Upload")
        ,
            mo.md(f"""
                Would you like to use sample data? {use_sample_data_switch}
            """)
        ,
            upload_block(use_sample_data_switch.value)
        ])
    )
    return


@app.cell(hide_code=True)
def _(fitz, mo, pdf_file, use_sample_data_switch):
    use_sample_data = use_sample_data_switch.value
    pdf_filename = "/content/sample_data/your_document.pdf"

    mo.output.replace(mo.status.spinner(title="Loading..."))
    print(pdf_file.value)
    if not use_sample_data and len(pdf_file.value) > 0:
        pdf_filename = pdf_file.value[0].name

    # Sample company data (represents info NOT in AI training data)
    sample_document = """ACME SaaS Company Internal Policies - Updated January 2025

    REFUND POLICY:
    - Standard plan: Full refund within 14 days, no questions asked
    - Pro plan: Full refund within 30 days of purchase
    - Enterprise: Custom refund terms per contract, minimum 60 days
    - Processing time: 3-5 business days via original payment method
    - Partial refunds available for annual subscriptions after initial period

    SUPPORT RESPONSE TIMES:
    - Starter plan: 72 hours response time, business days only
    - Professional plan: 24 hours response time, includes weekends
    - Enterprise plan: 4 hours response time, 24/7 coverage
    - Emergency issues: 2 hours for Enterprise, 8 hours for Professional
    - Critical system outages: 1 hour response for all paid plans

    BILLING AND PAYMENTS:
    - Invoices generated on the 1st of each month
    - Payment due within 15 days of invoice date
    - Late fee structure: $50 after 15 days, then $25 per additional week
    - Failed payments: Account suspended after 30 days, deleted after 90 days
    - Billing disputes must be raised within 60 days of invoice

    DATA AND SECURITY:
    - Active accounts: Data retained indefinitely with daily backups
    - Cancelled accounts: 90 days retention, Enterprise gets 180 days
    - Two-factor authentication: Required for admin accounts, optional for users
    - Password requirements: Minimum 12 characters with special characters
    - Session timeout: 30 minutes inactivity for regular users, 15 minutes for admins

    INTEGRATION FEATURES:
    - API rate limits: 1000 requests/hour Starter, 5000/hour Pro, unlimited Enterprise
    - Webhook support: Pro and Enterprise plans only
    - Custom SSO integration: Enterprise plan exclusive feature
    - Data export formats: CSV and JSON for all plans, XML for Enterprise only

    PRICING STRUCTURE:
    - Starter plan: $29/month, up to 5 users, basic features only
    - Professional plan: $99/month, up to 25 users, includes API access
    - Enterprise plan: Custom pricing, unlimited users, dedicated support
    - Setup fees: $500 for Enterprise, waived for annual contracts
    - Overage charges: $5 per additional user per month across all plans"""

    # Validation questions with known correct answers
    validation_questions = {
        "What is the refund period for Enterprise customers?": {
            "correct_answer": "minimum 60 days with custom terms per contract",
            "key_info": ["60 days", "custom terms", "contract", "Enterprise"]
        },
        "How quickly do you respond to Professional plan emergencies?": {
            "correct_answer": "8 hours for emergency issues",
            "key_info": ["8 hours", "Professional", "emergency"]
        },
        "What are the API rate limits for Pro plan?": {
            "correct_answer": "5000 requests per hour",
            "key_info": ["5000", "requests", "hour", "Pro"]
        },
        "When do admin sessions timeout?": {
            "correct_answer": "15 minutes of inactivity",
            "key_info": ["15 minutes", "admin", "timeout"]
        },
        "What is the setup fee for Enterprise plans?": {
            "correct_answer": "$500 setup fee, waived for annual contracts",
            "key_info": ["500", "setup fee", "Enterprise", "waived", "annual"]
        }
    }

    def extract_text_from_pdf(pdf_bytes):
        """Convert PDF to text for processing"""
        try:
            doc = fitz.open("pdf", pdf_bytes)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return None

    output_stack = []
    colour = "success"

    # Load data based on selection
    if use_sample_data:
        company_document = sample_document
        output_stack.append(mo.md(f"""
        :white_check_mark: Using sample company data (represents your internal company info)

        Document size: {len(company_document)} characters (~{len(company_document.split())} words)
        """))
    else:
        if len(pdf_file.value) > 0:
            output_stack.append(f"Processing PDF: {pdf_file.value[0].name}")
            pdf_text = extract_text_from_pdf(pdf_file.value[0].contents)
            if pdf_text:
                company_document = pdf_text
                validation_questions = {}  # Clear validation for custom PDF
                output_stack.append(mo.md(f":white_check_mark: PDF '{pdf_file.value[0].name}' loaded: {len(pdf_text)} characters (~{len(pdf_text.split())} words)"))
            else:
                colour = "danger"
                output_stack.append(mo.md(":x: Failed to extract text from PDF. Using sample data."))
                company_document = sample_document
        else:
            output_stack.append(f"""
            PDF missing. Using sample data.

            To use PDF: 1) Upload your file first
            """)
            company_document = sample_document

    output_stack.append(mo.md(f"""
    MEMORY CHALLENGE:

    - Document has ~{len(company_document.split())} words
    - AI Memory Limit: ~1,000 words at a time
    - The Problem: AI cannot 'see' the entire document at once!
    """))

    mo.callout(mo.vstack(output_stack),kind=colour)

    return company_document, validation_questions


@app.cell
def _():
    ### Strategy 1: Brute Force Memory (Context Stuffing)
    return


@app.cell
def _(mo, stg1_tab, stg2_tab, stg3_tab, stg4_tab):
    part_a_tabs = mo.ui.tabs(
        {
            "Strategy 1": stg1_tab,
            "Strategy 2": stg2_tab,
            "Strategy 3": stg3_tab,
            "Strategy 4": stg4_tab,
        }
    )
    mo.lazy(part_a_tabs,show_loading_indicator=True)
    return


@app.cell
def _(mo, model, question):
    stg1_params = mo.callout(mo.md(f"""
        ### Strategy 1 - Settings

        Select model to be used in test: {model}

        Question to be used for the test:

        {question}
    """))
    return (stg1_params,)


@app.cell
def _(client, model, question, time):
    # Strategy 1 - Functions

    model_to_test = model.value
    test_question = question.value

    def brute_force_memory(question, document, model):
        """Strategy 1: Load entire document into AI memory every time"""

        prompt = f"""You are a customer support assistant. Answer based only on the provided company information.

    COMPANY INFORMATION (LOADED INTO MEMORY):
    {document}

    CUSTOMER QUESTION: {question}

    Provide a specific answer based only on the company information above."""

        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        end_time = time.time()

        return response, (end_time - start_time)
    return brute_force_memory, model_to_test, test_question


@app.cell
def _(
    brute_force_memory,
    check_answer_accuracy,
    company_document,
    model_to_test,
    print_performance_metrics,
    test_question,
    validation_questions,
):
    # Strategy 1 - Outputs

    response, elapsed = brute_force_memory(test_question, company_document, model_to_test)

    is_correct = check_answer_accuracy(test_question, response.choices[0].message.content, validation_questions)
    result1 = print_performance_metrics(
        response, elapsed, "Strategy 1: Brute Force Memory",
        "Full document loaded", is_correct, "Load ENTIRE document into AI memory every single time", model_to_test, test_question
    )

    stg1_results = result1["mo_output"]
    return response, stg1_results


@app.cell
def _(mo, stg1_params, stg1_results):
    # Strategy 1 - Tab

    stg1_intro = mo.md(f"""
        ### Strategy 1: Brute Force Memory (Context Stuffing)
        **Test Parameters:** Put entire document in AI's working memory every time
    """)

    stg1_tab = mo.vstack([stg1_intro,stg1_params,stg1_results], gap=0)
    return (stg1_tab,)


@app.cell
def _():
    ### Strategy 2: Smart Memory Search (Keywords)
    return


@app.cell
def _(mo, model, number_of_chunks, question, size_words):
    stg2_params = mo.callout(mo.md(f"""
        ### Strategy 2 - Settings

        Select model to be used in test: {model}

        Question to be used for the test:

        {question}

        Number of words per chunck to be added to context: **{size_words.value}**
        {size_words}

        Number of chuncks to be added to context: **{number_of_chunks.value}**
        {number_of_chunks}
    """))
    return (stg2_params,)


@app.cell
def _(
    check_answer_accuracy,
    client,
    company_document,
    mo,
    model_to_test,
    number_of_chunks,
    print_performance_metrics,
    size_words,
    test_question,
    time,
    validation_questions,
):
    chunk_size_words = size_words.value
    chunks_to_retrieve = number_of_chunks.value

    def create_chunks(document, chunk_size=300):
        """Break document into smaller pieces for memory management"""
        words = document.split()
        chunks = []

        overlap = chunk_size // 4  # 25% overlap to avoid splitting related info
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())

        return chunks

    def keyword_smart_memory(question, document, model, chunk_size, top_k):
        """Strategy 2: Search for relevant pieces, only load those into AI memory"""

        start_time = time.time()

        # Step 1: Break document into manageable pieces
        chunks = create_chunks(document, chunk_size)

        # Step 2: Find pieces that match the question keywords
        question_words = set(question.lower().split())

        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words.intersection(chunk_words))
            score = overlap / len(question_words) if question_words else 0
            chunk_scores.append((score, i, chunk))

        # Step 3: Select top relevant pieces
        chunk_scores.sort(reverse=True)
        top_chunks = [(chunk, score) for score, _, chunk in chunk_scores[:top_k]]

        # Show what was selected
        mo_output_extension = [mo.md(f"**Memory Search Results (keyword matching):**")]
        for i, (chunk, score) in enumerate(top_chunks):
            mo_output_extension.append(mo.md(f"- Piece {i+1} (relevance: {score:.2f}): {chunk[:80]}..."))

        # Step 4: Load only relevant pieces into AI memory
        context = "\n\n".join([chunk for chunk, _ in top_chunks])

        prompt = f"""You are a customer support assistant. Answer based only on the provided information.

    RELEVANT INFORMATION (LOADED INTO MEMORY):
    {context}

    CUSTOMER QUESTION: {question}

    Provide a specific answer based only on the information above."""

        response_1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        end_time = time.time()

        return response_1, (end_time - start_time), len(chunks), len(top_chunks), mo_output_extension

    response_1, elapsed_1, total_chunks, used_chunks, mo_output_extension = keyword_smart_memory(
        test_question, company_document, model_to_test, chunk_size_words, chunks_to_retrieve
    )

    is_correct_1 = check_answer_accuracy(test_question, response_1.choices[0].message.content, validation_questions)
    result2 = print_performance_metrics(
        response_1, elapsed_1, "Strategy 2: Smart Memory Search (Keywords)",
        f"{used_chunks}/{total_chunks} pieces", is_correct_1, "Find relevant pieces first, only load those into memory", model_to_test, test_question, mo_output_extension
    )
    stg2_results = result2["mo_output"]
    return chunk_size_words, chunks_to_retrieve, create_chunks, stg2_results


@app.cell
def _(mo, stg2_params, stg2_results):
    # Strategy 2 - Tab

    stg2_intro = mo.md(f"""
        ### Strategy 2
        **Idea:** Use keywords/regex to pull only relevant slices, then prompt.  
        **Pros:** Cheap and fast; good for structured wording.  
        **Cons:** Misses synonyms / paraphrase; brittle to phrasing.
    """)

    stg2_tab = mo.vstack([stg2_intro,stg2_params,stg2_results], gap=0)
    return (stg2_tab,)


@app.cell
def _():
    ### Strategy 3: Semantic Memory Search (Understanding)
    return


@app.cell
def _(mo, model, number_of_chunks, question, size_words):
    stg3_params = mo.callout(mo.md(f"""
        ### Strategy 3 - Settings

        Select model to be used in test: {model}

        Question to be used for the test:

        {question}

        Number of words per chunck to be added to context: **{size_words.value}**
        {size_words}

        Number of chuncks to be added to context: **{number_of_chunks.value}**
        {number_of_chunks}
    """))
    return (stg3_params,)


@app.cell
def _(
    check_answer_accuracy,
    chunk_size_words,
    chunks_to_retrieve,
    client,
    company_document,
    create_chunks,
    embedding_model,
    mo,
    model,
    model_to_test,
    np,
    print_performance_metrics,
    response,
    test_question,
    time,
    validation_questions,
):
    def semantic_smart_memory(question, document, model, chunk_size, top_k, embedding_model):
        """Strategy 3: AI understands meaning, not just keyword matching"""
        start_time = time.time()
        chunks = create_chunks(document, chunk_size)
        print('🧠 AI analyzing document meaning (this may take a moment)...')
        chunk_embeddings = embedding_model.encode(chunks)
        query_embedding = embedding_model.encode([question])
        similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [(chunks[i], similarities[i]) for i in top_indices]
        mo_output_extension = [mo.md(f'**🔍 Semantic Memory Search Results (meaning-based):**')]
        for (i, (chunk, score)) in enumerate(top_chunks):
            mo_output_extension.append(mo.md(f'- Piece {i + 1} (meaning similarity: {score:.3f}): {chunk[:80]}...'))
        context = '\n\n'.join([chunk for (chunk, _) in top_chunks])
        prompt = f'You are a customer support assistant. Answer based only on the provided information.\n\nRELEVANT INFORMATION (LOADED INTO MEMORY):\n{context}\n\nCUSTOMER QUESTION: {question}\n\nProvide a specific answer based only on the information above.'
        response_2 = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}])
        end_time = time.time()
        return (response, end_time - start_time, len(chunks), len(top_chunks), mo_output_extension)
    (response_2, elapsed_2, total_chunks_1, used_chunks_1, mo_output_extension_3) = semantic_smart_memory(test_question, company_document, model_to_test, chunk_size_words, chunks_to_retrieve, embedding_model)
    is_correct_2 = check_answer_accuracy(test_question, response_2.choices[0].message.content, validation_questions)
    result3 = print_performance_metrics(response_2, elapsed_2, 'Strategy 3: Semantic Memory Search (Understanding)', f'{used_chunks_1}/{total_chunks_1} pieces', is_correct_2, "AI understands MEANING, finds relevant info even with different words",model, test_question, mo_output_extension_3)
    stg3_results = result3["mo_output"]
    return (stg3_results,)


@app.cell
def _(mo, stg3_params, stg3_results):
    # Strategy 3 - Tab

    stg3_intro = mo.md(f"""
        ### Strategy 3: Semantic Memory Search (Understanding-Based RAG)
        **Idea:** Use embeddings to find semantically similar chunks (not just keyword matches).  
        **Pros:** Robust to paraphrase; better recall.  
        **Cons:** Needs a vector index; can surface near-matches that are **contextually wrong** if chunking is poor.
    """)

    stg3_tab = mo.vstack([stg3_intro,stg3_params,stg3_results], gap=0)
    return (stg3_tab,)


@app.cell
def _():
    ### Strategy 4: Hybrid Memory + AI Double-Check
    return


@app.cell
def _(
    ai_reranking,
    final_chunks,
    initial_chunks,
    mean_slider,
    mo,
    model,
    question,
    size_words,
):
    stg4_params = mo.callout(mo.md(f"""
        ### Strategy 4 - Settings

        Select model to be used in test: {model}

        Question to be used for the test:
        {question}

        Number of words per chunck to be added to context: **{size_words.value}**
        {size_words}

        Number of chuncks to be added to context: **{initial_chunks.value}**
        {initial_chunks}

        Final chunks: **{final_chunks.value}**
        {final_chunks}

        Use AI ReRanking: {ai_reranking}

        Meaning Weight: **{mean_slider.value}**
        {mean_slider}
    """))
    return (stg4_params,)


@app.cell
def _(
    ai_reranking,
    check_answer_accuracy,
    chunk_size_words,
    client,
    company_document,
    create_chunks,
    embedding_model,
    final_chunks,
    initial_chunks,
    mo,
    model,
    model_to_test,
    np,
    print_performance_metrics,
    test_question,
    time,
    validation_questions,
):
    initial_chunks_to_find = initial_chunks.value
    final_chunks_to_use = final_chunks.value
    use_ai_reranking = ai_reranking.value
    meaning_weight = 0.7

    def hybrid_smart_memory(question, document, model, chunk_size, initial_top_k, final_top_k, embedding_model, use_rerank=True, meaning_weight=0.7):
        """Strategy 4: Combine keyword + meaning search, then AI double-checks"""
        start_time = time.time()
        chunks = create_chunks(document, chunk_size)
        print('🧠 Combining meaning analysis + keyword matching...')
        chunk_embeddings = embedding_model.encode(chunks)
        query_embedding = embedding_model.encode([question])
        meaning_scores = np.dot(query_embedding, chunk_embeddings.T)[0]
        question_words = set(question.lower().split())
        keyword_scores = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words.intersection(chunk_words))
            score = overlap / len(question_words) if question_words else 0
            keyword_scores.append(score)
        keyword_weight = 1.0 - meaning_weight
        hybrid_scores = []
        for i in range(len(chunks)):
            combined_score = meaning_weight * meaning_scores[i] + keyword_weight * keyword_scores[i]
            hybrid_scores.append((combined_score, i, chunks[i]))
        hybrid_scores.sort(reverse=True)
        initial_chunks = [chunk for (_, _, chunk) in hybrid_scores[:initial_top_k]]
        mo_output_extension = [mo.md(f'**🔍 Initial Selection ({initial_top_k} pieces, hybrid scoring):**')]
        for (i, (score, _, chunk)) in enumerate(hybrid_scores[:initial_top_k]):
            mo_output_extension.append(mo.md(f'- Piece {i + 1} (hybrid score: {score:.3f}): {chunk[:80]}...'))
        final_chunks = initial_chunks
        rerank_time = 0
        if use_rerank and len(initial_chunks) > final_top_k:
            print(f'\n🤖 AI Double-Check: Asking AI to rank which pieces are most relevant...')
            rerank_start = time.time()
            chunk_list = ''
            for (i, chunk) in enumerate(initial_chunks, 1):
                chunk_list = chunk_list + f'\n{i}. {chunk}\n'
            rerank_prompt = f'Given this question: "{question}"\n\nRank these information pieces from most relevant (1) to least relevant for answering the question.\n\nPIECES TO RANK:\n{chunk_list}\n\nRespond with only the numbers in order of relevance (e.g., "3,1,2,4").'
            rerank_response = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': rerank_prompt}])
            rerank_end = time.time()
            rerank_time = rerank_end - rerank_start
            try:
                ranking = [int(x.strip()) for x in rerank_response.choices[0].message.content.split(',')]
                final_chunks = [initial_chunks[i - 1] for i in ranking[:final_top_k]]
                print(f'✓ AI ranked pieces: {rerank_response.choices[0].message.content}')
            except:
                print('⚠ AI ranking failed, using original order')
                final_chunks = initial_chunks[:final_top_k]
        else:
            final_chunks = initial_chunks[:final_top_k]
        mo_output_extension.append(mo.md(f'**📋 Final Selection:** {len(final_chunks)} pieces loaded into memory'))
        context = '\n\n'.join(final_chunks)
        prompt = f'You are a customer support assistant. Answer based only on the provided information.\n\nRELEVANT INFORMATION (LOADED INTO MEMORY):\n{context}\n\nCUSTOMER QUESTION: {question}\n\nProvide a specific answer based only on the information above.'
        response = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}])
        end_time = time.time()
        total_time = end_time - start_time
        return (response, total_time, len(chunks), len(final_chunks), rerank_time, mo_output_extension)
    (response_3, elapsed_3, total_chunks_2, used_chunks_2, rerank_time, mo_output_extension_4) = hybrid_smart_memory(test_question, company_document, model_to_test, chunk_size_words, initial_chunks_to_find, final_chunks_to_use, embedding_model, use_ai_reranking, meaning_weight)
    mo_output_extension_4.insert(0,mo.md(f'**Balance:** {meaning_weight:.0%} meaning, {1 - meaning_weight:.0%} keywords'))
    is_correct_3 = check_answer_accuracy(test_question, response_3.choices[0].message.content, validation_questions)
    chunks_info = f'{used_chunks_2}/{total_chunks_2} pieces'
    if use_ai_reranking:
        chunks_info = chunks_info + f', AI rerank: +{rerank_time:.2f}s'
    result4 = print_performance_metrics(response_3, elapsed_3, 'Strategy 4: Hybrid + AI Double-Check', chunks_info, is_correct_3, "Combine keyword + meaning search, then AI ranks the best pieces", model, test_question, mo_output_extension_4)
    stg4_results = result4["mo_output"]
    return (stg4_results,)


@app.cell
def _(mo, stg4_params, stg4_results):
    # Strategy 4 - Tab

    stg4_intro = mo.md(f"""
        ### Strategy 4: Hybrid Memory + AI Double-Check

        **Idea:** Retrieve (keyword + semantic), then ask the model to **cross-check** and cite evidence.  
        **Pros:** Better guard against errors; encourages quoting sources.  
        **Cons:** Extra passes → more latency/cost.
    """)

    stg4_tab = mo.vstack([stg4_intro,stg4_params,stg4_results], gap=0)
    return (stg4_tab,)


@app.cell
def _():
    print("""
    YOUR ANALYSIS WORKSHEET
    ==========================

    For each strategy you tested, record your findings:

    STRATEGY 1 - BRUTE FORCE MEMORY (Context Stuffing):
    Memory Used: _____ tokens | Response Time: _____ seconds | Accuracy: ✓/✗
    Notes: _________________________________________________

    STRATEGY 2 - KEYWORD MEMORY SEARCH:
    Memory Used: _____ tokens | Response Time: _____ seconds | Accuracy: ✓/✗
    Notes: _________________________________________________

    STRATEGY 3 - SEMANTIC MEMORY SEARCH (Understanding):
    Memory Used: _____ tokens | Response Time: _____ seconds | Accuracy: ✓/✗
    Notes: _________________________________________________

    STRATEGY 4 - HYBRID + AI DOUBLE-CHECK:
    Memory Used: _____ tokens | Response Time: _____ seconds | Accuracy: ✓/✗
    AI Rerank Time: _____ seconds
    Notes: _________________________________________________

    MODEL PERFORMANCE PATTERNS (fill in based on your tests):

    SPEED RANKING: 1.______ 2.______ 3.______ 4.______
    ACCURACY RANKING: 1.______ 2.______ 3.______ 4.______
    MEMORY EFFICIENCY: 1.______ 2.______ 3.______ 4.______

    YOUR BUSINESS DECISIONS:

    1. Which strategy would you choose for your use case? Why?
       ___________________________________________________

    2. What trade-offs are you willing to make?
       ___________________________________________________

    3. Which model performed best for YOUR needs?
       ___________________________________________________

    4. What will you implement first when you return to your company?
       ___________________________________________________

    🚀 NEXT STEPS FOR YOUR COMPANY:
    • Test with your actual company documents
    • Try questions your real customers ask
    • Calculate memory costs for your expected volume
    • Choose strategy based on your accuracy vs speed vs cost priorities
    """)
    return


@app.cell
def _(mo):
    ### Parameters Declaration
    # mo.output.replace(mo.md("### Part A - Parameters Declaration"))

    # Part A

    # Company Data
    use_sample_data_switch = mo.ui.switch(value=True)
    pdf_file = mo.ui.file(filetypes=[".pdf"])

    # For all strategies
    model = mo.ui.dropdown(options=["openai/gpt-oss-20b:free","meta-llama/llama-3.3-70b-instruct:free","google/gemini-2.0-flash-exp:free","qwen/qwen2.5-vl-72b-instruct:free","qwen/qwen3-coder:free","moonshotai/kimi-k2:free", "deepseek/deepseek-chat-v3.1:free", "z-ai/glm-4.5-air:free", "deepseek/deepseek-r1:free", "tngtech/deepseek-r1t-chimera:free"], value="openai/gpt-oss-20b:free")
    question = mo.ui.text(value="What are the API rate limits for Pro plan?", full_width=True)

    # Strategy 2 to 4
    size_words = mo.ui.slider(start=200, stop=600, step=50, value=300, full_width=True, show_value=True)

    # Strategy 2 and 3
    number_of_chunks = mo.ui.slider(start=1, stop=4, step=1, value=2, full_width=True, show_value=True)

    # Strategy 4
    initial_chunks = mo.ui.slider(start=3, stop=6, step=1, value=4, full_width=True, show_value=True)
    final_chunks = mo.ui.slider(start=1, stop=3, step=1, value=2, full_width=True, show_value=True)
    ai_reranking = mo.ui.switch(value=True)
    mean_slider = mo.ui.slider(start=0.1, stop=0.9, step=0.1, value=0.7, full_width=True, show_value=True)
    return (
        ai_reranking,
        final_chunks,
        initial_chunks,
        mean_slider,
        model,
        number_of_chunks,
        pdf_file,
        question,
        size_words,
        use_sample_data_switch,
    )


if __name__ == "__main__":
    app.run()
