# 2. PRETRAGA WEBA (TAVILY API) - Za opšta pitanja
        if self.tavily_api_key and not is_internal_query:
            try:
                tavily_resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key, 
                        "query": user_question, 
                        "search_depth": "basic", 
                        "include_answer": True, 
                        "max_results": 2
                    },
                    timeout=10
                )
                print(f"👉 Tavily Status Code: {tavily_resp.status_code}")
                
                if tavily_resp.status_code == 200:
                    data = tavily_resp.json()
                    tavily_answer = data.get("answer", "")
                    if not tavily_answer:
                        snippets = [res["content"] for res in data.get("results", [])]
                        tavily_answer = " ".join(snippets)
                    
                    if tavily_answer:
                        context_texts.append(f"Izvor: Web Pretraga (Tavily)\n{tavily_answer}")
                        if "Web Pretraga (Tavily)" not in sources:
                            sources.append("Web Pretraga (Tavily)")
                else:
                    print(f"❌ Tavily greška odgovora: {tavily_resp.text}")
            except Exception as e:
                print(f"❌ Tavily izuzetak (konekcija/timeout): {e}")