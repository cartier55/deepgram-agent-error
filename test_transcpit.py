import asyncio
from dotenv import load_dotenv
import os
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone
from custom_types import ResponseRequiredRequest, Utterance
# from llm_client_phased import LlmClient as PhasedLlmClient
from llm_client import LlmClient
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv()
console = Console()
API_KEY = os.getenv("DG_API_KEY")

class ConversationTracker:
    def __init__(self):
        self.utterances = []
        self.response_id = 0

    def add_utterance(self, role, content):
        self.utterances.append(Utterance(role=role, content=content))
        if role == "user":
            self.response_id += 1

async def process_transcription(transcript, conversation_tracker, llm_client):
    if transcript.strip():
        console.print(f"Speaker: {transcript}")
        conversation_tracker.add_utterance("user", transcript)
        request = ResponseRequiredRequest(
            interaction_type="response_required",
            response_id=conversation_tracker.response_id,
            transcript=conversation_tracker.utterances
        )
        full_response = ""
        with console.status("[bold blue]AI is thinking...", spinner="dots"):
            async for response in llm_client.draft_response(request):
                full_response += response.content
                if response.content_complete:
                    conversation_tracker.add_utterance("agent", full_response)
        console.print(Panel(Text(full_response, style="bold blue"), title="AI", expand=False))

async def run_transcription(llm_client):
    conversation_tracker = ConversationTracker()
    deepgram = DeepgramClient(API_KEY)
    
    async def on_message(_, result):
        transcript = result.channel.alternatives[0].transcript
        if result.speech_final:
            await process_transcription(transcript, conversation_tracker, llm_client)

    try:
        connection = deepgram.listen.asyncwebsocket.v("1")
        # connection = deepgram.listen.asynclive.v("1")

        connection.on(LiveTranscriptionEvents.Transcript, on_message)

        await connection.start(LiveOptions(
        # connection.start(LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=True
        ))

        
        microphone = Microphone(connection.send)
        microphone.start()

        print("Listening... Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        if 'microphone' in locals():
            microphone.finish()
        if 'connection' in locals():
            await connection.finish()
            # connection.finish()

async def main():
    llm_client = LlmClient(call_id="voice_chat_session")
    print(llm_client.draft_begin_message())

    try:
        await run_transcription(llm_client)
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")

if __name__ == "__main__":
    asyncio.run(main())





