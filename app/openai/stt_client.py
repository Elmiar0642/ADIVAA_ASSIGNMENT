class STTClient:
    async def transcribe_chunk(self, audio_chunk: bytes) -> str:
        if not audio_chunk:
            return ""
        return f"transcribed {len(audio_chunk)} bytes of audio"

