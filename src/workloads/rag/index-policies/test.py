import requests

# Simulate a customer conversation
conversations = [
    # Premium customer, damaged phone, COD, wants cash
    "I ordered an iPhone from you, paid by cash on delivery, but the screen is cracked. I want my money back in cash immediately. What do I do?",
    
    # Customer wanting cancellation after shipment
    "I placed an order yesterday but it already shipped. Can I cancel it? Will there be any charges?",
    
    # Warranty confusion - customer thinks physical damage is covered
    "I dropped my phone and the screen shattered. It's still under warranty. How do I get it fixed for free?",
    
    # Escalation scenario
    "I returned a product 10 days ago and still haven't received my refund. Your support team isn't responding. I want to escalate this.",
    
    # Ambiguous query - delivery delay but also wants to cancel
    "My order was supposed to arrive 3 days ago. At this point I just want to cancel and get my money back. It was paid by UPI."
]

for query in conversations:
    # Get embedding
    r = requests.post("http://localhost:8200/embed", json={"texts": [query]})
    vec = r.json()["vectors"][0]
    
    # Search
    r = requests.post(
        "http://localhost:6333/collections/kestral_policies/points/search",
        json={"vector": vec, "limit": 5, "with_payload": True}
    )
    results = r.json()["result"]
    
    print(f"\n{'='*80}")
    print(f"CUSTOMER: {query}")
    print(f"{'='*80}")
    print("RETRIEVED POLICIES (for LLM to synthesize):")
    for i, hit in enumerate(results):
        p = hit["payload"]
        text_preview = p['text'][:200].replace('\n', ' ')
        print(f"\n  [{p['policy_name']}] {p['heading_path']}")
        print(f"  Score: {hit['score']:.3f} | Tags: {p.get('tags', [])}")
        print(f"  Content: {text_preview}...")