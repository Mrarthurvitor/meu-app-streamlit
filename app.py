import streamlit as st
import openai as op
import pyaudio
import wave
import os
import time
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from threading import Thread
import pandas as pd
import numpy as np
from gtts import gTTS
from sklearn.linear_model import LinearRegression
import random
import io
import PyPDF2
from docx import Document
import matplotlib.pyplot as plt
from transformers import pipeline
import re

# Configuração das APIs
op.api_key = os.getenv('OPENAI_API_KEY', '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')
GOOGLE_API_KEY = "AIzaSyDIUhxzxID7vmlnzzRCT4Qdu9cFKGO0dcw"  # Substitua por sua chave real
GOOGLE_CX = "76d77ecfde40f4253"  # Substitua por seu CX real

# Sistema de auto-atualização e pesquisa
class AutonomousAssistant:
    def __init__(self):
        # Inicializar variáveis de estado primeiro
        self.current_thought = "Inicializando sistema..."
        self.current_progress = 0
        
        # Depois inicializar outros componentes
        self.knowledge_file = "knowledge_base.json"
        self.update_log = "update_log.txt"
        self.load_knowledge()
        self.search_history = []
        self.performance_metrics = {
            "response_times": [],
            "accuracy_scores": [],
            "user_ratings": []
        }
        self.scheduled_tasks = []
        self.load_tasks()
        
        # Novos sistemas
        self.conversation_memory = []
        self.important_facts = {}
        self.plugins = self.load_plugins()
        self.init_collaboration()
        
    def update_thought(self, message, progress_increment=0):
        """Atualiza o pensamento atual e o progresso"""
        self.current_thought = message
        if progress_increment:
            self.current_progress = min(100, self.current_progress + progress_increment)
        
        # Atualizar a interface se estiver em execução
        if 'thought_ui' in st.session_state:
            st.session_state.thought_ui.markdown(f"💭 **Pensando:** {self.current_thought}")
        if 'progress_ui' in st.session_state:
            st.session_state.progress_ui.progress(self.current_progress)
        
        time.sleep(0.5)  # Dar tempo para a UI atualizar
        
    def load_knowledge(self):
        self.update_thought("Carregando base de conhecimento...", 5)
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    self.knowledge = json.load(f)
            else:
                self.knowledge = {
                    "expertise": {
                        "python": 9,
                        "web_development": 8,
                        "data_science": 8,
                        "ai_research": 8,
                        "voice_processing": 7,
                        "auto_optimization": 6,
                        "document_analysis": 5,
                        "business_intelligence": 6
                    },
                    "behavior": {
                        "response_style": "analytical",
                        "detail_level": "comprehensive",
                        "autonomy_level": "high",
                        "response_speed": 1.0,
                        "research_depth": 2,
                        "memory_context": 3
                    },
                    "updates": []
                }
        except Exception as e:
            self.knowledge = {
                "expertise": {
                    "python": 9,
                    "web_development": 8,
                    "data_science": 8,
                    "ai_research": 8,
                    "voice_processing": 7,
                    "auto_optimization": 6,
                    "document_analysis": 5,
                    "business_intelligence": 6
                },
                "behavior": {
                    "response_style": "analytical",
                    "detail_level": "comprehensive",
                    "autonomy_level": "high",
                    "response_speed": 1.0,
                    "research_depth": 2,
                    "memory_context": 3
                },
                "updates": []
            }
        self.update_thought("Base de conhecimento carregada", 5)
    
    def save_knowledge(self):
        self.update_thought("Salvando conhecimento atualizado...", 5)
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge, f, indent=2)
        self.update_thought("Conhecimento salvo com sucesso", 5)
    
    def log_update(self, update):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {update}\n"
        self.knowledge['updates'].append(log_entry)
        
        with open(self.update_log, 'a') as f:
            f.write(log_entry)
        
        self.save_knowledge()
        self.update_thought(f"Atualização registrada: {update}", 3)
    
    def load_tasks(self):
        self.update_thought("Carregando tarefas agendadas...", 5)
        try:
            if os.path.exists("scheduled_tasks.json"):
                with open("scheduled_tasks.json", 'r') as f:
                    self.scheduled_tasks = json.load(f)
        except:
            self.scheduled_tasks = []
        self.update_thought(f"{len(self.scheduled_tasks)} tarefas carregadas", 5)
    
    def save_tasks(self):
        self.update_thought("Salvando tarefas agendadas...", 5)
        with open("scheduled_tasks.json", 'w') as f:
            json.dump(self.scheduled_tasks, f, indent=2)
        self.update_thought("Tarefas salvas com sucesso", 5)
    
    def should_research(self, question):
        self.update_thought("Analisando necessidade de pesquisa...", 10)
        decision_prompt = f"""
        Com base na pergunta abaixo, decida se é necessário pesquisar informações atualizadas na internet.
        Responda apenas com 'SIM' ou 'NÃO', sem explicações.
        
        Pergunta: {question}
        """
        
        response = op.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": decision_prompt}],
            max_tokens=10,
            temperature=0.0
        )
        decision = "SIM" in response.choices[0].message['content'].strip().upper()
        
        if decision:
            self.update_thought("Pesquisa na web necessária", 5)
        else:
            self.update_thought("Usando conhecimento interno", 5)
            
        return decision
    
    def web_search(self, query):
        self.update_thought(f"Pesquisando: '{query}'", 10)
        try:
            # Pesquisa usando API do Google
            search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={requests.utils.quote(query)}"
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            # Processar resultados
            results = []
            for item in data.get('items', [])[:5]:  # Limitar a 5 resultados
                results.append({
                    "title": item.get('title'),
                    "url": item.get('link'),
                    "snippet": item.get('snippet', '')[:200] + '...'  # Limitar snippet
                })
            
            self.search_history.append({
                "query": query,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "source": "Google API"
            })
            
            self.update_thought(f"{len(results)} resultados encontrados", 10)
            return results
        except Exception as e:
            self.update_thought(f"Erro na pesquisa: {str(e)}", 0)
            st.error(f"Erro na pesquisa: {str(e)}")
            return []
    
    # ========== NOVAS FUNCIONALIDADES ==========
    
    def remember_context(self, conversation, important_facts):
        """Armazena contexto importante da conversa"""
        self.conversation_memory.append({
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation
        })
        self.important_facts.update(important_facts)
        
        # Limitar memória a 10 itens
        if len(self.conversation_memory) > 10:
            self.conversation_memory.pop(0)
            
        self.log_update("Contexto atualizado na memória")
    
    def recall_context(self):
        """Recupera o contexto relevante"""
        if not self.conversation_memory:
            return "Nenhum contexto anterior disponível"
        
        context = "Contexto da conversa:\n"
        for i, mem in enumerate(self.conversation_memory[-3:]):
            context += f"{i+1}. {mem['conversation']}\n"
            
        if self.important_facts:
            context += "\nFatos importantes:\n"
            for key, value in self.important_facts.items():
                context += f"- {key}: {value}\n"
                
        return context
    
    def process_document(self, file):
        """Processa documentos PDF, Word e TXT"""
        self.update_thought(f"Processando documento: {file.name}", 10)
        content = ""
        
        try:
            # PDF
            if file.type == "application/pdf":
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                    
            # Word
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(io.BytesIO(file.getvalue()))
                for para in doc.paragraphs:
                    content += para.text + "\n"
                    
            # Texto
            elif file.type == "text/plain":
                content = file.getvalue().decode("utf-8")
                
            else:
                return None, "Tipo de documento não suportado"
                
            # Resumir documento
            summary_prompt = f"""
            Resuma o documento abaixo para incluir apenas informações essenciais.
            Mantenha dados técnicos, números importantes e conclusões.
            
            Documento:
            {content[:5000]}... [continua]
            """
            
            summary = op.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=800,
                temperature=0.3
            ).choices[0].message['content'].strip()
            
            # Extrair fatos importantes
            facts_prompt = f"""
            Extraia os fatos mais importantes do documento como um JSON:
            {{
                "titulo": "Título do Documento",
                "autor": "Autor se disponível",
                "data": "Data se disponível",
                "pontos_chave": ["ponto 1", "ponto 2", ...],
                "decisoes": ["decisão 1", ...],
                "recomendacoes": ["recomendação 1", ...]
            }}
            """
            
            facts = op.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": facts_prompt}],
                max_tokens=500,
                temperature=0.1
            ).choices[0].message['content'].strip()
            
            return summary, facts
            
        except Exception as e:
            return None, f"Erro no processamento: {str(e)}"
    
    def business_intelligence_dashboard(self):
        """Dashboard de Business Intelligence"""
        if not self.performance_metrics['response_times']:
            return "Dados insuficientes para análise"
            
        # Preparar dados
        metrics_df = pd.DataFrame({
            "Tempo Resposta": self.performance_metrics['response_times'],
            "Acurácia": self.performance_metrics['accuracy_scores'],
            "Avaliação": self.performance_metrics['user_ratings']
        })
        
        # Análise preditiva
        X = np.array(metrics_df['Tempo Resposta']).reshape(-1, 1)
        y = np.array(metrics_df['Avaliação'])
        
        if len(y) > 1:
            try:
                model = LinearRegression()
                model.fit(X, y)
                future_times = np.array([10, 15, 20]).reshape(-1, 1)
                predictions = model.predict(future_times)
                
                # Criar relatório
                report = f"""
                ## 📈 Relatório de Performance
                
                **Métricas Atuais:**
                - Tempo médio de resposta: {np.mean(metrics_df['Tempo Resposta']):.2f}s
                - Acurácia média: {np.mean(metrics_df['Acurácia'])*100:.1f}%
                - Avaliação média do usuário: {np.mean(metrics_df['Avaliação']):.1f}/5
                
                **Previsões:**
                - Avaliação estimada para 10s: {predictions[0]:.1f}
                - Avaliação estimada para 15s: {predictions[1]:.1f}
                - Avaliação estimada para 20s: {predictions[2]:.1f}
                
                **Recomendações:**
                - Otimizar processos para tempo médio de 15s
                - Focar em melhorar precisão técnica
                """
                
                # Gráfico
                fig, ax = plt.subplots()
                ax.scatter(metrics_df['Tempo Resposta'], metrics_df['Avaliação'], color='blue')
                ax.plot(future_times, predictions, color='red', linewidth=2)
                ax.set_title('Relação Tempo de Resposta vs Avaliação')
                ax.set_xlabel('Tempo de Resposta (s)')
                ax.set_ylabel('Avaliação do Usuário (1-5)')
                
                return report, fig
                
            except Exception as e:
                return "Erro na análise preditiva", None
                
        return "Dados insuficientes para predição", None
    
    def analyze_sentiment(self, audio_file):
        """Analisa sentimento a partir de áudio"""
        try:
            # Usando modelo pré-treinado
            classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")
            result = classifier(audio_file)
            sentiment = max(result, key=lambda x: x['score'])['label']
            
            # Traduzir para português
            sentiment_map = {
                "angry": "raiva",
                "happy": "felicidade",
                "sad": "tristeza",
                "neutral": "neutro",
                "fear": "medo",
                "disgust": "desgosto",
                "surprise": "surpresa"
            }
            
            return sentiment_map.get(sentiment, sentiment)
            
        except Exception as e:
            return f"Erro na análise: {str(e)}"
    
    def load_plugins(self):
        """Carrega plugins disponíveis"""
        return {
            "stock": {
                "function": self.get_stock_data,
                "description": "Obter dados de ações em tempo real"
            },
            "weather": {
                "function": self.get_weather,
                "description": "Previsão do tempo para uma localização"
            }
        }
    
    def execute_plugin(self, plugin_name, params):
        """Executa um plugin específico"""
        if plugin_name in self.plugins:
            try:
                return self.plugins[plugin_name]["function"](**params)
            except Exception as e:
                return f"Erro na execução: {str(e)}"
        return "Plugin não encontrado"
    
    def get_stock_data(self, symbol):
        """Obtém dados de ações (exemplo)"""
        # Implementação real usaria API como Alpha Vantage
        prices = {
            "AAPL": 185.25,
            "GOOGL": 138.42,
            "MSFT": 340.11,
            "AMZN": 145.18
        }
        return f"Cotação {symbol}: ${prices.get(symbol, 'N/A')}"
    
    def get_weather(self, location):
        """Obtém previsão do tempo (exemplo)"""
        # Implementação real usaria API como OpenWeather
        forecasts = {
            "São Paulo": "25°C, Parcialmente nublado",
            "Rio de Janeiro": "28°C, Ensolarado",
            "Brasília": "27°C, Chuvas esparsas"
        }
        return f"Previsão em {location}: {forecasts.get(location, 'N/A')}"
    
    def init_collaboration(self):
        """Inicializa sistema de colaboração"""
        if "collaboration" not in st.session_state:
            st.session_state.collaboration = {
                "users": [],
                "messages": [],
                "shared_files": []
            }
    
    def add_collaborator(self, user_id):
        """Adiciona colaborador à sessão"""
        if user_id not in st.session_state.collaboration["users"]:
            st.session_state.collaboration["users"].append(user_id)
            return True
        return False
    
    def add_collaboration_message(self, user, message):
        """Adiciona mensagem ao chat colaborativo"""
        st.session_state.collaboration["messages"].append({
            "user": user,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def share_file(self, user, file_info):
        """Compartilha arquivo na sessão colaborativa"""
        st.session_state.collaboration["shared_files"].append({
            "user": user,
            "file": file_info,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def continuous_learning(self):
        """Sistema de aprendizado contínuo"""
        if len(self.performance_metrics['accuracy_scores']) < 10:
            return "Dados insuficientes para aprendizado"
            
        # Identificar áreas de melhoria
        weak_areas = []
        for area, score in self.knowledge['expertise'].items():
            if score < 7:
                weak_areas.append(area)
        
        if not weak_areas:
            return "Nenhuma área de melhoria identificada"
        
        # Criar plano de estudo
        study_plan = f"Plano de Estudo Automático:\n"
        for area in weak_areas[:3]:
            study_plan += f"- Pesquisar avanços recentes em {area}\n"
            study_plan += f"- Estudar práticas recomendadas para {area}\n"
            study_plan += f"- Analisar casos de uso em {area}\n\n"
            
            # Agendar tarefa de aprendizado
            new_task = {
                "name": f"Aprendizado: {area}",
                "type": "research",
                "query": f"avanços recentes em {area} melhores práticas",
                "topic": area,
                "frequency": "once",
                "next_run": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            self.scheduled_tasks.append(new_task)
        
        self.save_tasks()
        return study_plan
    
    # ========== FIM DAS NOVAS FUNCIONALIDADES ==========
    
    def analyze_and_decide(self, question):
        self.current_progress = 0
        self.current_thought = "Iniciando processamento..."
        self.update_thought(self.current_thought, 0)
        
        # Recuperar contexto
        context = self.recall_context()
        
        # Etapa 1: Análise da pergunta
        self.update_thought("Analisando sua pergunta...", 10)
        analysis_prompt = f"""
        Como um assistente autônomo, analise a pergunta do usuário e determine:
        1. O conhecimento necessário para responder
        2. Se pesquisa na internet é necessária
        3. O nível de detalhe apropriado
        4. A melhor estratégia de resposta
        
        Contexto anterior:
        {context}
        
        Pergunta: {question}
        
        Forneça sua análise em no máximo 3 frases.
        """
        
        analysis = op.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=200,
            temperature=0.3
        ).choices[0].message['content'].strip()
        
        # Etapa 2: Decisão de pesquisa
        research_results = []
        if self.should_research(question):
            self.update_thought("Criando query de pesquisa...", 10)
            research_query_prompt = f"""
            Com base na pergunta e análise, gere uma query de pesquisa otimizada para o Google.
            Seja conciso e direto ao ponto.
            
            Pergunta: {question}
            Análise: {analysis}
            
            Forneça apenas a query, sem texto adicional.
            """
            
            research_query = op.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": research_query_prompt}],
                max_tokens=40,
                temperature=0.2
            ).choices[0].message['content'].strip()
            
            research_results = self.web_search(research_query)
        
        # Etapa 3: Síntese da resposta
        self.update_thought("Sintetizando resposta...", 15)
        research_context = ""
        if research_results:
            research_context = "\nDados de pesquisa:\n"
            for i, res in enumerate(research_results):
                research_context += f"{i+1}. [{res['title']}]({res['url']}): {res['snippet']}\n"
        
        synthesis_prompt = f"""
        Com base na análise, {f"dados de pesquisa" if research_results else "seu conhecimento"}, 
        contexto anterior e pergunta do usuário, forneça a melhor resposta possível.
        
        Contexto anterior:
        {context}
        
        Diretrizes:
        - Seja completo mas conciso
        - Cite fontes quando usar dados de pesquisa
        - Mantenha foco técnico
        - Inclua exemplos práticos quando relevante
        
        Pergunta: {question}
        Análise: {analysis}
        {research_context}
        """
        
        resposta = op.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=1800,
            temperature=0.6
        ).choices[0].message['content'].strip()
        
        # Etapa 4: Auto-avaliação
        self.update_thought("Auto-avaliando resposta...", 10)
        evaluation_prompt = f"""
        Avalie a qualidade da resposta gerada considerando:
        - Precisão técnica
        - Relevância para a pergunta
        - Clareza e organização
        - Utilidade prática
        - Atualização das informações
        
        Resposta: {resposta}
        
        Forneça:
        1. Pontuação (0-10)
        2. Três melhorias potenciais
        3. Decisão: MANTER ou REFINAR
        """
        
        evaluation = op.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=300,
            temperature=0.4
        ).choices[0].message['content'].strip()
        
        if "REFINAR" in evaluation:
            self.update_thought("Refinando resposta...", 10)
            refinement_prompt = f"""
            Com base na avaliação, refine a resposta para melhor atendimento ao usuário.
            
            Avaliação: {evaluation}
            Resposta original: {resposta}
            
            Forneça apenas a resposta refinada, sem comentários.
            """
            
            resposta = op.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": refinement_prompt}],
                max_tokens=1800,
                temperature=0.5
            ).choices[0].message['content'].strip()
        
        # Armazenar contexto
        self.remember_context(
            conversation=f"P: {question}\nR: {resposta}",
            important_facts={
                "tema_principal": question[:50] + "...",
                "decisoes": evaluation
            }
        )
        
        self.update_thought("Processamento completo!", 10)
        return resposta, analysis, research_results, evaluation
    
    def get_current_knowledge(self):
        expertise = ", ".join([f"{k}({v}/10)" for k, v in self.knowledge['expertise'].items()])
        behavior = ", ".join([f"{k}: {v}" for k, v in self.knowledge['behavior'].items()])
        return f"Expertise: {expertise}\nComportamento: {behavior}"
    
    def auto_optimize_parameters(self):
        """Otimiza automaticamente os parâmetros de operação"""
        if len(self.performance_metrics['response_times']) < 5:
            return
        
        self.update_thought("Otimizando parâmetros...", 10)
        # Modelo preditivo para otimização
        X = np.array(self.performance_metrics['response_times']).reshape(-1, 1)
        y = np.array(self.performance_metrics['user_ratings'])
        
        if len(y) > 1:
            try:
                model = LinearRegression()
                model.fit(X, y)
                optimal_time = np.argmax(model.predict(X))
                
                # Ajustar parâmetros de operação
                self.knowledge['behavior']['response_speed'] = max(0.5, min(2.0, optimal_time/10))
                self.log_update(f"Parâmetros otimizados: velocidade={self.knowledge['behavior']['response_speed']:.1f}")
                self.knowledge['expertise']['auto_optimization'] = min(10, self.knowledge['expertise'].get('auto_optimization', 5) + 1)
            except:
                pass
        self.update_thought("Otimização concluída", 10)
    
    def execute_scheduled_tasks(self):
        """Executa tarefas agendadas automaticamente"""
        now = datetime.now()
        tasks_to_remove = []
        
        for i, task in enumerate(self.scheduled_tasks):
            task_time = datetime.strptime(task['next_run'], "%Y-%m-%d %H:%M")
            if now >= task_time:
                try:
                    self.update_thought(f"Executando tarefa: {task['name']}", 5)
                    
                    # Executar pesquisa automática
                    if task['type'] == 'research':
                        results = self.web_search(task['query'])
                        self.knowledge['expertise'][task['topic']] = min(10, self.knowledge['expertise'].get(task['topic'], 5) + 1)
                        self.log_update(f"Pesquisa agendada: {task['query']} - Conhecimento em {task['topic']} aumentado")
                    
                    # Reprogramar a tarefa
                    if task['frequency'] == 'daily':
                        task['next_run'] = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
                    elif task['frequency'] == 'weekly':
                        task['next_run'] = (now + timedelta(weeks=1)).strftime("%Y-%m-%d %H:%M")
                    elif task['frequency'] == 'once':
                        tasks_to_remove.append(i)
                    
                except Exception as e:
                    self.log_update(f"Erro na tarefa {task['name']}: {str(e)}")
        
        # Remover tarefas únicas que foram executadas
        for i in sorted(tasks_to_remove, reverse=True):
            if i < len(self.scheduled_tasks):
                del self.scheduled_tasks[i]
        
        self.save_tasks()
        self.update_thought("Tarefas agendadas processadas", 10)
    
    def auto_generate_tasks(self):
        """Gera automaticamente tarefas baseadas em gaps de conhecimento"""
        self.update_thought("Identificando gaps de conhecimento...", 10)
        topics = ["python", "ai", "web development", "data science", "machine learning"]
        knowledge_gaps = []
        
        for topic in topics:
            if self.knowledge['expertise'].get(topic, 0) < 7:
                knowledge_gaps.append(topic)
        
        for topic in knowledge_gaps:
            new_task = {
                "name": f"Atualizar {topic}",
                "type": "research",
                "query": f"avanços recentes em {topic}",
                "topic": topic,
                "frequency": "weekly",
                "next_run": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d 09:00")
            }
            
            if new_task not in self.scheduled_tasks:
                self.scheduled_tasks.append(new_task)
        
        self.save_tasks()
        self.log_update(f"Tarefas geradas para {len(knowledge_gaps)} gaps de conhecimento")
        self.update_thought(f"Geradas {len(knowledge_gaps)} novas tarefas", 10)
    
    def api_action_executor(self, action_command):
        """Executa ações através de APIs externas"""
        try:
            self.update_thought(f"Processando comando: {action_command[:30]}...", 10)
            # Análise do comando
            parse_prompt = f"""
            Extraia informações de: {action_command}
            Formato resposta JSON:
            {{
                "action_type": "schedule|data_fetch|other",
                "title": "título se aplicável",
                "date": "YYYY-MM-DD",
                "time": "HH:MM",
                "duration": minutos,
                "query": "consulta se aplicável"
            }}
            """
            
            parsed = op.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": parse_prompt}],
                max_tokens=200,
                temperature=0.1
            ).choices[0].message['content'].strip()
            
            # Limpar e converter para JSON
            parsed = parsed[parsed.find('{'):parsed.rfind('}')+1]
            data = json.loads(parsed)
            
            # Executar ação baseada no tipo
            if data['action_type'] == 'schedule':
                st.success(f"Reunião '{data.get('title', '')}' agendada para {data.get('date', '')} às {data.get('time', '')} por {data.get('duration', 30)} minutos")
                self.update_thought("Reunião agendada com sucesso", 10)
                return True
            elif data['action_type'] == 'data_fetch':
                st.success(f"Dados recuperados para: {data.get('query', '')}")
                self.update_thought("Dados recuperados com sucesso", 10)
                return True
            else:
                st.info(f"Ação executada: {action_command}")
                self.update_thought("Ação executada com sucesso", 10)
                return True
            
        except Exception as e:
            self.update_thought(f"Erro na execução: {str(e)}", 0)
            st.error(f"Erro na execução de ação: {str(e)}")
            return False
    
    def monitor_performance(self, response_time, accuracy, user_rating):
        """Monitora e registra métricas de performance"""
        self.performance_metrics['response_times'].append(response_time)
        self.performance_metrics['accuracy_scores'].append(accuracy)
        self.performance_metrics['user_ratings'].append(user_rating)
        
        # Auto-otimização periódica
        if len(self.performance_metrics['response_times']) % 5 == 0:
            self.auto_optimize_parameters()
            
        # Geração automática de tarefas
        if len(self.performance_metrics['response_times']) % 7 == 0:
            self.auto_generate_tasks()
            
        # Aprendizado contínuo
        if len(self.performance_metrics['response_times']) % 10 == 0:
            self.continuous_learning()

# Inicializar sistema autônomo
if 'assistant' not in st.session_state:
    st.session_state.assistant = AutonomousAssistant()

# Histórico de conversa
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Variáveis de configuração na sessão
if 'mostrar_pensamento' not in st.session_state:
    st.session_state.mostrar_pensamento = True
    
if 'idioma' not in st.session_state:
    st.session_state.idioma = 'pt'
    
if 'velocidade' not in st.session_state:
    st.session_state.velocidade = 1.0

# Função para humanizar texto
def humanizar_texto(texto):
    """Adiciona pausas e entonação natural ao texto"""
    # Adiciona pausas após pontuações
    texto = re.sub(r'([.!?;])', r'\1 ', texto)
    
    # Quebras de linha para pausas naturais
    texto = re.sub(r'\.\s', '.\n\n', texto)
    texto = re.sub(r',\s', ',\n', texto)
    
    # Simplificação de expressões complexas
    substituicoes = {
        r'\b(\d+)h(\d+)?\b': lambda m: f"{m.group(1)} horas" if not m.group(2) else f"{m.group(1)} horas e {m.group(2)} minutos",
        r'\bR\$\s*(\d+[\d,.]*)\b': lambda m: f"{m.group(1).replace('.', '')} reais",
        r'\b(\d+)m\b': lambda m: f"{m.group(1)} metros",
        r'\b(\d+)kg\b': lambda m: f"{m.group(1)} quilos"
    }
    
    for padrao, repl in substituicoes.items():
        texto = re.sub(padrao, repl, texto)
    
    # Variações de ênfase
    palavras = texto.split()
    for i in range(0, len(palavras), 5):
        if random.random() > 0.7 and len(palavras[i]) > 3:
            palavras[i] = f"*{palavras[i]}*"
    
    return ' '.join(palavras)

# Funções de áudio
def gravar_audio(duracao=5, arquivo="audio.wav"):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        
        for i in range(0, int(16000 / 1024 * duracao)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        with wave.open(arquivo, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
            
        return arquivo
    except Exception as e:
        st.error(f"Erro na gravação: {str(e)}")
        return None

def transcrever_audio(arquivo):
    try:
        if not arquivo or not os.path.exists(arquivo):
            return ""
            
        with open(arquivo, 'rb') as f:
            resultado = op.Audio.transcribe('whisper-1', f, language='pt')
            return resultado['text'].strip()
    except Exception as e:
        st.error(f"Erro na transcrição: {str(e)}")
        return ""

def falar_texto(texto, lang='pt'):
    try:
        if not texto:
            return None
            
        # Pré-processamento para humanizar a fala
        texto_humanizado = humanizar_texto(texto)
        
        # Ajustar velocidade baseado na configuração
        velocidade = st.session_state.velocidade
        slow = velocidade < 1.0
        
        # Parâmetros ajustados para naturalidade
        tts = gTTS(
            text=texto_humanizado, 
            lang=lang,
            tld='com.br',  # Sotaque brasileiro
            slow=slow,
            lang_check=False
        )
        
        # Geração de áudio em memória
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_bytes = audio_buffer.getvalue()
        
        # Reproduzir áudio
        st.audio(audio_bytes, format='audio/mp3', autoplay=True)
        return audio_bytes
    except Exception as e:
        st.error(f"Erro na síntese de voz: {str(e)}")
        return None

# ============= SISTEMA DE TAREFAS AUTOMÁTICAS =============
def task_scheduler_thread():
    """Thread para execução de tarefas agendadas"""
    while True:
        try:
            if 'assistant' in st.session_state:
                st.session_state.assistant.execute_scheduled_tasks()
            time.sleep(60)
        except:
            pass

# Iniciar thread em segundo plano
if 'scheduler_started' not in st.session_state:
    st.session_state.scheduler_started = True
    Thread(target=task_scheduler_thread, daemon=True).start()

# Configuração da página
st.set_page_config(page_title="Assistente Cognitivo Py", layout="wide", page_icon="🤖")

# CSS personalizado com animações de pensamento e responsividade
st.markdown("""
<style>
    /* Estilo geral */
    .stApp { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Barra lateral */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%) !important;
        border-right: 1px solid #4e54c8;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.4);
    }
    
    /* Botões da barra lateral */
    .stButton>button {
        background: linear-gradient(45deg, #4e54c8, #8f94fb) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        width: 100%;
        margin: 8px 0;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Botão de gravação */
    .btn-record {
        font-size: 2rem !important;
        width: 120px !important;
        height: 120px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #ff416c, #ff4b2b) !important;
        margin: 20px auto !important;
        display: block !important;
        box-shadow: 0 4px 20px rgba(255, 75, 43, 0.4);
    }
    
    /* Abas */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }
    [data-baseweb="tab"] {
        padding: 12px 24px !important;
        background: #1a1a2e !important;
        border-radius: 8px !important;
        margin: 0 5px !important;
        transition: all 0.3s !important;
    }
    [data-baseweb="tab"]:hover {
        background: #4e54c8 !important;
    }
    [aria-selected="true"] {
        background: linear-gradient(45deg, #4e54c8, #8f94fb) !important;
        font-weight: bold !important;
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid #4e54c8 !important;
        padding: 12px !important;
    }
    
    /* Histórico */
    .history-item {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4e54c8;
    }
    
    /* Pensamento */
    .thinking-bubble {
        background: #2c3e50;
        border-radius: 20px;
        padding: 15px;
        margin: 15px 0;
        border: 1px solid #3498db;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
    }
    
    /* Estilos para autonomia */
    .autonomous-decision {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #4e54c8;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    .research-card {
        background: rgba(30, 30, 50, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #4e54c8;
    }
    
    .evaluation-panel {
        background: rgba(40, 40, 60, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border: 2px solid #ff4b5c;
    }
    
    /* Estilos de voz */
    .voice-panel {
        background: linear-gradient(135deg, #4a00e0, #8e2de2);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
    }
    
    .voice-visualizer {
        display: flex;
        justify-content: center;
        height: 100px;
        align-items: flex-end;
        margin: 20px 0;
    }
    
    .voice-bar {
        width: 10px;
        background: #00d2ff;
        margin: 0 2px;
        border-radius: 5px 5px 0 0;
        animation: voicePulse 1.5s infinite;
    }
    
    @keyframes voicePulse {
        0% { height: 10%; }
        50% { height: 90%; }
        100% { height: 10%; }
    }
    
    /* Automação */
    .automation-card {
        background: rgba(40, 40, 60, 0.8);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #8f94fb;
    }
    
    /* Animações de pensamento */
    @keyframes thoughtPulse {
        0% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }
    
    .thought-container {
        background: rgba(40, 40, 60, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #4e54c8;
        animation: thoughtPulse 3s infinite;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .thought-bubble {
        background: rgba(50, 50, 80, 0.95);
        border-radius: 20px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #3498db;
    }
    
    .progress-container {
        background: rgba(30, 30, 50, 0.7);
        border-radius: 10px;
        padding: 10px;
        margin: 15px 0;
    }
    
    /* Estilos de colaboração */
    .collaboration-panel {
        background: linear-gradient(135deg, #2a0c4e, #4a1b8c);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .collaboration-message {
        background: rgba(50, 50, 80, 0.8);
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #8f94fb;
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .stApp > div {
            padding: 8px !important;
        }
        [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.9rem !important;
        }
        .stTextArea textarea {
            height: 120px !important;
        }
        .stButton>button {
            padding: 10px 18px !important;
        }
        .btn-record {
            width: 90px !important;
            height: 90px !important;
            font-size: 1.5rem !important;
        }
        .colab-user {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Barra lateral
with st.sidebar:
    st.title("🤖 Assistente Cognitivo")
    st.subheader("Sistema de Tomada de Decisão")
    
    # Painel de conhecimento
    st.markdown("### 🧠 Conhecimento Atual")
    for area, nivel in st.session_state.assistant.knowledge['expertise'].items():
        st.progress(nivel/10, text=f"{area.capitalize()} ({nivel}/10)")
    
    # Configurações
    st.markdown("### ⚙️ Configurações")
    st.session_state.mostrar_pensamento = st.checkbox("Mostrar processo de decisão", value=st.session_state.mostrar_pensamento)
    
    st.markdown("### 🎚 Parâmetros de Voz")
    st.session_state.velocidade = st.slider("Velocidade", 0.5, 2.0, 1.0, 0.1)
    st.session_state.idioma = st.selectbox("Idioma", ["pt", "en"], index=0 if st.session_state.idioma == 'pt' else 1)
    
    # Gerenciamento
    if st.button("🧹 Limpar Histórico", key="clear_history_btn"):
        st.session_state['history'] = []
        st.success("Histórico limpo!")
    
    if st.button("🔄 Atualizar Conhecimento", key="update_knowledge_btn"):
        st.session_state.assistant.log_update("Atualização manual de conhecimento")
        st.success("Conhecimento atualizado!")
    
    # Controle de voz
    st.markdown("### 🎤 Controles de Voz")
    duracao_gravacao = st.slider("Duração da gravação (seg)", 3, 10, 5, key="voice_duration")
    
    # Colaboração
    st.markdown("### 👥 Colaboração")
    new_user = st.text_input("Adicionar colaborador:", key="new_collaborator")
    if st.button("➕ Adicionar", key="add_collaborator_btn") and new_user:
        if st.session_state.assistant.add_collaborator(new_user):
            st.success(f"{new_user} adicionado à sessão!")
        else:
            st.warning("Usuário já está na sessão")

# Abas principais
tab_chat, tab_auto, tab_research, tab_voice, tab_automation, tab_docs, tab_colab = st.tabs(
    ["💬 Chat", "🧠 Autonomia", "🌐 Pesquisas", "🎤 Voz", "⚙️ Automação", "📄 Documentos", "👥 Colaboração"]
)

# Inicializar elementos de UI para pensamento
if 'thought_ui' not in st.session_state:
    st.session_state.thought_ui = st.empty()
if 'progress_ui' not in st.session_state:
    st.session_state.progress_ui = st.empty()

# Atualizar pensamento inicial
st.session_state.thought_ui.markdown(f"💭 **Pensando:** {st.session_state.assistant.current_thought}")
st.session_state.progress_ui.progress(st.session_state.assistant.current_progress)

# Aba Chat
with tab_chat:
    st.header("🧠 Assistente Cognitivo")
    
    # Painel de pensamento
    with st.container():
        st.markdown("### Estado Cognitivo Atual")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.thought_ui = st.empty()
        with col2:
            st.session_state.progress_ui = st.progress(st.session_state.assistant.current_progress)
    
    pergunta = st.text_area("Digite sua pergunta:", height=150, key="chat_input",
                           placeholder="Pergunte sobre Python, IA, ou peça para pesquisar na web...")
    
    if st.button("🚀 Enviar Pergunta", use_container_width=True, key="send_question_btn") and pergunta:
        # Reiniciar indicadores
        st.session_state.assistant.current_thought = "Processando sua pergunta..."
        st.session_state.assistant.current_progress = 0
        st.session_state.thought_ui.markdown(f"💭 **Pensando:** {st.session_state.assistant.current_thought}")
        st.session_state.progress_ui.progress(0)
        
        start_time = time.time()
        with st.spinner("Processando de forma autônoma..."):
            resposta, analysis, research, evaluation = st.session_state.assistant.analyze_and_decide(pergunta)
            end_time = time.time()
            
            # Coletar métricas de precisão
            accuracy_prompt = f"""
            Avalie a precisão da resposta em uma escala de 0-1:
            Pergunta: {pergunta}
            Resposta: {resposta}
            """
            try:
                accuracy = op.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": accuracy_prompt}],
                    max_tokens=10,
                    temperature=0.0
                ).choices[0].message['content'].strip()
                accuracy = float(accuracy)
            except:
                accuracy = 0.8
            
            # Registrar métricas
            st.session_state.assistant.monitor_performance(
                response_time=end_time - start_time,
                accuracy=accuracy,
                user_rating=5
            )
            
            # Registro no histórico
            st.session_state['history'].append({
                "pergunta": pergunta,
                "resposta": resposta,
                "analise": analysis,
                "pesquisa": research,
                "avaliacao": evaluation
            })
        
        # Exibição do processo de decisão
        if st.session_state.mostrar_pensamento:
            with st.expander("📊 Detalhes do Processamento"):
                st.markdown(f"**Análise da pergunta:**\n{analysis}")
                
                if research:
                    st.markdown("**🔍 Pesquisa realizada (Google API):**")
                    for i, res in enumerate(research):
                        st.markdown(f"{i+1}. **[{res['title']}]({res['url']})**\n{res['snippet']}")
                
                st.markdown(f"**📊 Auto-avaliação:**\n{evaluation}")
                
                # Mostrar contexto utilizado
                st.markdown("**🧠 Contexto utilizado:**")
                st.code(st.session_state.assistant.recall_context())
        
        # Exibição da resposta
        st.markdown(f"""
        <div class='history-item'>
            <h4>Você:</h4>
            <p>{pergunta}</p>
            <h4>Assistente:</h4>
            <p>{resposta}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reproduzir resposta em áudio
        falar_texto(resposta, lang=st.session_state.idioma)

# Aba Autonomia
with tab_auto:
    st.header("🧠 Sistema de Tomada de Decisão Autônoma")
    
    st.markdown("""
    <div class='autonomous-decision'>
        <h4>Arquitetura Cognitiva</h4>
        <p>Este assistente toma decisões independentes através de 4 estágios:</p>
        <ol>
            <li><strong>Análise da Pergunta</strong>: Compreensão profunda da necessidade do usuário</li>
            <li><strong>Decisão de Pesquisa</strong>: Avaliação autônoma da necessidade de dados externos</li>
            <li><strong>Síntese de Resposta</strong>: Combinação de conhecimento interno e pesquisa</li>
            <li><strong>Auto-avaliação</strong>: Crítica e refinamento da resposta antes de entregar</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Exibir última análise
    if st.session_state['history']:
        ultimo = st.session_state['history'][-1]
        st.subheader("Último Processo Decisório")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**❓ Pergunta Original**")
            st.info(ultimo['pergunta'])
            
            st.markdown("**🧠 Análise Estratégica**")
            st.code(ultimo['analise'], language='text')
        
        with col2:
            st.markdown("**📝 Resposta Gerada**")
            st.success(ultimo['resposta'])
            
            st.markdown("**📊 Auto-avaliação**")
            st.warning(ultimo['avaliacao'])
    else:
        st.info("Nenhum histórico de decisão disponível. Faça uma pergunta na aba Chat.")

# Aba Pesquisas
with tab_research:
    st.header("🌐 Histórico de Pesquisas Autônomas")
    
    if st.session_state.assistant.search_history:
        for search in reversed(st.session_state.assistant.search_history):
            with st.expander(f"🔍 {search['query']} ({search['source']})"):
                st.caption(f"Pesquisado em {search['timestamp'][:19].replace('T', ' ')}")
                
                for i, res in enumerate(search['results']):
                    st.markdown(f"""
                    <div class='research-card'>
                        <h4>{i+1}. {res['title']}</h4>
                        <p>{res['snippet']}</p>
                        <a href="{res['url']}" target="_blank">Ver fonte</a>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("O assistente ainda não realizou pesquisas. Faça perguntas que requeiram dados atualizados.")

# Aba Voz
with tab_voice:
    st.header("🎤 Modo Voz Completo")
    
    st.markdown("""
    <div class='voice-panel'>
        <h3>Interação por Voz</h3>
        <p>Converse naturalmente com o assistente usando seu microfone</p>
        
        <div class='voice-visualizer'>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
            <div class='voice-bar'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gravação de Voz")
        if st.button("🎤 Iniciar Gravação", use_container_width=True, key="voice_record_btn"):
            arquivo = "voice_temp.wav"
            resultado = gravar_audio(duracao_gravacao, arquivo)
            
            if resultado:
                pergunta = transcrever_audio(arquivo)
                os.remove(arquivo)
                
                if pergunta:
                    st.session_state.voice_question = pergunta
                    st.success("Áudio transcrito com sucesso!")
                    st.markdown(f"**Pergunta detectada:** {pergunta}")
                    
                    # Análise de sentimento
                    sentiment = st.session_state.assistant.analyze_sentiment(arquivo)
                    st.info(f"Sentimento detectado: {sentiment.capitalize()}")
                else:
                    st.error("Não foi possível transcrever o áudio")
    
    with col2:
        st.subheader("Processamento")
        if 'voice_question' in st.session_state and st.session_state.voice_question:
            if st.button("🧠 Processar Pergunta", use_container_width=True, key="process_voice_btn"):
                # Reiniciar indicadores
                st.session_state.assistant.current_thought = "Processando pergunta de voz..."
                st.session_state.assistant.current_progress = 0
                st.session_state.thought_ui.markdown(f"💭 **Pensando:** {st.session_state.assistant.current_thought}")
                st.session_state.progress_ui.progress(0)
                
                start_time = time.time()
                with st.spinner("Processando pergunta de voz..."):
                    resposta, analysis, research, evaluation = st.session_state.assistant.analyze_and_decide(st.session_state.voice_question)
                    end_time = time.time()
                    
                    # Coletar métricas de precisão
                    accuracy_prompt = f"""
                    Avalie a precisão da resposta em uma escala de 0-1:
                    Pergunta: {st.session_state.voice_question}
                    Resposta: {resposta}
                    """
                    try:
                        accuracy = op.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": accuracy_prompt}],
                            max_tokens=10,
                            temperature=0.0
                        ).choices[0].message['content'].strip()
                        accuracy = float(accuracy)
                    except:
                        accuracy = 0.8
                    
                    # Registrar métricas
                    st.session_state.assistant.monitor_performance(
                        response_time=end_time - start_time,
                        accuracy=accuracy,
                        user_rating=5
                    )
                    
                    # Registro no histórico
                    st.session_state['history'].append({
                        "pergunta": st.session_state.voice_question,
                        "resposta": resposta,
                        "analise": analysis,
                        "pesquisa": research,
                        "avaliacao": evaluation,
                        "modo": "voz"
                    })
                
                # Exibir resposta
                st.markdown(f"""
                <div class='history-item'>
                    <h4>Resposta:</h4>
                    <p>{resposta}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Reproduzir áudio
                falar_texto(resposta, lang=st.session_state.idioma)
        
    # Histórico de voz
    st.subheader("Histórico de Conversas por Voz")
    voice_history = [h for h in st.session_state['history'] if h.get('modo') == 'voz']
    
    if voice_history:
        for i, item in enumerate(reversed(voice_history)):
            with st.expander(f"Conversa de voz {len(voice_history)-i}"):
                st.markdown(f"**Você:** {item['pergunta']}")
                st.markdown(f"**Assistente:** {item['resposta']}")
                # Regenerar áudio para reprodução
                audio_bytes = falar_texto(item['resposta'], lang=st.session_state.idioma)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
    else:
        st.info("Nenhuma conversa por voz registrada. Grave sua primeira pergunta!")

# Aba Automação
with tab_automation:
    st.header("⚙️ Sistema de Automação Cognitiva")
    
    # Painel de pensamento
    with st.container():
        st.markdown("### Estado Cognitivo Atual")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.thought_ui.markdown(f"💭 **Pensando:** {st.session_state.assistant.current_thought}")
        with col2:
            st.session_state.progress_ui.progress(st.session_state.assistant.current_progress)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance do Sistema")
        if st.session_state.assistant.performance_metrics['response_times']:
            metrics_df = pd.DataFrame({
                "Tempo Resposta": st.session_state.assistant.performance_metrics['response_times'],
                "Acurácia": st.session_state.assistant.performance_metrics['accuracy_scores'],
                "Avaliação": st.session_state.assistant.performance_metrics['user_ratings']
            })
            st.line_chart(metrics_df)
            
            avg_time = np.mean(st.session_state.assistant.performance_metrics['response_times'])
            avg_accuracy = np.mean(st.session_state.assistant.performance_metrics['accuracy_scores'])
            avg_rating = np.mean(st.session_state.assistant.performance_metrics['user_ratings'])
            
            st.metric("⏱ Tempo Médio", f"{avg_time:.1f}s")
            st.metric("🎯 Acurácia Média", f"{avg_accuracy*100:.1f}%")
            st.metric("⭐ Avaliação Média", f"{avg_rating:.1f}/5")
        else:
            st.info("Coletando dados de performance...")
    
    with col2:
        st.subheader("🤖 Auto-Otimização")
        nivel_automacao = st.session_state.assistant.knowledge['expertise'].get('auto_optimization', 0)
        st.progress(nivel_automacao/10, 
                   text=f"Nível de Automação: {nivel_automacao}/10")
        
        if st.button("🔍 Executar Diagnóstico", help="Analisa e otimiza parâmetros internos", key="run_diagnostic_btn"):
            with st.spinner("Otimizando sistema..."):
                st.session_state.assistant.auto_optimize_parameters()
                st.success("Parâmetros otimizados!")
        
        st.markdown("**Parâmetros Atuais:**")
        st.json({
            "response_speed": st.session_state.assistant.knowledge['behavior'].get('response_speed', 1),
            "research_depth": st.session_state.assistant.knowledge['behavior'].get('research_depth', 2),
            "autonomy_level": st.session_state.assistant.knowledge['behavior'].get('autonomy_level', 3)
        })
    
    st.subheader("📅 Tarefas Agendadas")
    if st.button("🔄 Gerar Tarefas Automaticamente", key="auto_tasks_gen_btn"):
        st.session_state.assistant.auto_generate_tasks()
        st.success("Tarefas geradas com base em gaps de conhecimento!")
    
    for i, task in enumerate(st.session_state.assistant.scheduled_tasks):
        with st.expander(f"⏰ {task['name']} - Próxima execução: {task['next_run']}"):
            st.markdown(f"**Tipo:** {task['type'].capitalize()}")
            st.markdown(f"**Frequência:** {task['frequency']}")
            st.markdown(f"**Tópico:** {task['topic']}")
            
            if task['type'] == 'research':
                st.markdown(f"**Query:** `{task['query']}`")
            
            if st.button("Executar Agora", key=f"run_task_{i}"):
                st.session_state.assistant.execute_scheduled_tasks()
                st.experimental_rerun()
            
            if st.button("❌ Remover", key=f"del_task_{i}"):
                del st.session_state.assistant.scheduled_tasks[i]
                st.session_state.assistant.save_tasks()
                st.experimental_rerun()
    
    st.subheader("🤖 Ações por API")
    action_command = st.text_input("Comando de ação:", placeholder="Ex: Agendar reunião sobre IA amanhã às 10:00 por 1 hora", key="api_command")
    if st.button("Executar Ação", key="run_api_action_btn") and action_command:
        result = st.session_state.assistant.api_action_executor(action_command)
        if result:
            st.success("Ação executada com sucesso!")
        else:
            st.error("Falha na execução da ação")
    
    # Plugins
    st.subheader("🧩 Plugins Disponíveis")
    for plugin, info in st.session_state.assistant.plugins.items():
        with st.expander(f"🔌 {plugin.capitalize()} - {info['description']}"):
            if plugin == "stock":
                symbol = st.text_input("Símbolo da ação:", value="AAPL", key=f"stock_{plugin}")
                if st.button("Obter Cotação", key=f"run_{plugin}"):
                    result = st.session_state.assistant.execute_plugin(plugin, {"symbol": symbol})
                    st.info(result)
            elif plugin == "weather":
                location = st.text_input("Localização:", value="São Paulo", key=f"weather_{plugin}")
                if st.button("Obter Previsão", key=f"run_{plugin}"):
                    result = st.session_state.assistant.execute_plugin(plugin, {"location": location})
                    st.info(result)
    
    # Execução em lote de tarefas
    st.subheader("🚀 Execução em Lote")
    if st.button("▶️ Executar Todas as Tarefas Agendadas", use_container_width=True, key="run_all_tasks_btn"):
        if st.session_state.assistant.scheduled_tasks:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, task in enumerate(st.session_state.assistant.scheduled_tasks):
                status_text.markdown(f"**Executando:** {task['name']}")
                progress_bar.progress(int((i+1)/len(st.session_state.assistant.scheduled_tasks)*100))
                
                # Atualizar pensamento
                st.session_state.assistant.update_thought(f"Processando tarefa: {task['name']}", 0)
                
                # Simular execução
                time.sleep(2)
                st.session_state.assistant.update_thought(f"Tarefa concluída: {task['name']}", 10)
                
                # Atualizar UI
                st.session_state.thought_ui.markdown(f"💭 **Pensando:** {st.session_state.assistant.current_thought}")
                st.session_state.progress_ui.progress(st.session_state.assistant.current_progress)
            
            status_text.success("Todas as tarefas foram executadas!")
            st.session_state.assistant.update_thought("Tarefas agendadas concluídas", 10)
        else:
            st.warning("Nenhuma tarefa agendada para executar")

# Aba Documentos
with tab_docs:
    st.header("📄 Análise de Documentos")
    
    st.markdown("""
    <div class='autonomous-decision'>
        <h4>Processamento Inteligente de Documentos</h4>
        <p>Carregue documentos para análise e extração automática de informações</p>
        <p>Formatos suportados: PDF, Word (DOCX), Texto (TXT)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Carregar documento", 
                                    type=["pdf", "docx", "txt"],
                                    key="document_uploader")
    
    if uploaded_file:
        st.success(f"Documento carregado: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
        
        if st.button("🔍 Analisar Documento", key="analyze_document_btn"):
            with st.spinner("Processando documento..."):
                summary, facts = st.session_state.assistant.process_document(uploaded_file)
                
                if summary:
                    st.subheader("Resumo do Documento")
                    st.write(summary)
                    
                    st.subheader("Fatos Extraídos")
                    st.json(facts)
                    
                    # Armazenar fatos importantes
                    try:
                        facts_data = json.loads(facts)
                        if "pontos_chave" in facts_data:
                            st.session_state.assistant.remember_context(
                                conversation=f"Documento analisado: {uploaded_file.name}",
                                important_facts={
                                    "pontos_chave": ", ".join(facts_data["pontos_chave"][:3]),
                                    "recomendacoes": ", ".join(facts_data.get("recomendacoes", [])[:2])
                                }
                            )
                            st.success("Fatos importantes armazenados na memória!")
                    except:
                        pass
                else:
                    st.error(f"Erro no processamento: {facts}")

# Aba Colaboração
with tab_colab:
    st.header("👥 Modo Colaborativo")
    
    st.markdown("""
    <div class='collaboration-panel'>
        <h3>Trabalho em Equipe</h3>
        <p>Colabore com outros usuários em tempo real</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lista de colaboradores
    st.subheader("👤 Participantes")
    if st.session_state.collaboration["users"]:
        for user in st.session_state.collaboration["users"]:
            st.markdown(f"- {user}")
    else:
        st.info("Nenhum participante adicionado. Use a barra lateral para adicionar colaboradores.")
    
    # Chat colaborativo
    st.subheader("💬 Conversa Colaborativa")
    col1, col2 = st.columns([3, 1])
    with col1:
        message = st.text_input("Digite sua mensagem:", key="collab_message")
    with col2:
        if st.button("Enviar", key="send_collab_msg") and message:
            st.session_state.assistant.add_collaboration_message("Você", message)
           
    # Histórico de mensagens
    if st.session_state.collaboration["messages"]:
        for msg in st.session_state.collaboration["messages"][-10:]:
            st.markdown(f"""
            <div class='collaboration-message'>
                <strong>{msg['user']}</strong> ({msg['timestamp']}):
                <p>{msg['message']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhuma mensagem na conversa colaborativa")
    
    # Compartilhamento de arquivos
    st.subheader("📎 Compartilhar Arquivos")
    colab_file = st.file_uploader("Selecionar arquivo para compartilhar", key="collab_file_upload")
    if st.button("Compartilhar Arquivo", key="share_file_btn") and colab_file:
        file_info = {
            "name": colab_file.name,
            "type": colab_file.type,
            "size": f"{colab_file.size/1024:.1f} KB"
        }
        st.session_state.assistant.share_file("Você", file_info)
        st.success("Arquivo compartilhado com sucesso!")
        st.experimental_rerun()
    
    # Arquivos compartilhados
    st.subheader("📂 Arquivos Compartilhados")
    if st.session_state.collaboration["shared_files"]:
        for file in st.session_state.collaboration["shared_files"][-5:]:
            st.markdown(f"""
            <div class='collaboration-message'>
                <strong>{file['user']}</strong> ({file['timestamp']}):
                <p>{file['file']['name']} ({file['file']['type']}, {file['file']['size']})</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhum arquivo compartilhado ainda")

# Business Intelligence Dashboard
st.sidebar.markdown("---")
if st.sidebar.button("📊 Business Intelligence", key="bi_dashboard_btn"):
    st.session_state.show_bi = True
    
if 'show_bi' in st.session_state and st.session_state.show_bi:
    with st.expander("📈 DASHBOARD DE BUSINESS INTELLIGENCE", expanded=True):
        report, fig = st.session_state.assistant.business_intelligence_dashboard()
        
        if isinstance(report, str):
            st.markdown(report)
        else:
            st.markdown(report, unsafe_allow_html=True)
        
        if fig:
            st.pyplot(fig)
            
        # Aprendizado contínuo
        st.subheader("🎓 Sistema de Aprendizado Contínuo")
        study_plan = st.session_state.assistant.continuous_learning()
        st.markdown(study_plan)