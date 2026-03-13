import asyncio
import json
from file_utilities import load_image, clear_images

class SignalConnector:
    
    def __init__(self, host='127.0.0.1', port=7583):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        clear_images()
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        await self.subscribe()
    
    async def subscribe(self):
        msg = {"jsonrpc": "2.0", "method": "subscribeReceive", "id": 1}
        self.writer.write(json.dumps(msg).encode() + b'\n')

    async def receive(self):
        line = await self.reader.readline()
        if not line:
            return None

        data = json.loads(line)
        print(f"RAW: {json.dumps(data, indent=2)}")

        if "method" in data and data["method"] == "receive":
            params = data.get('params', {})
            envelope = params.get('envelope', params)
            source = envelope.get('source', envelope.get('sourceNumber', ''))
            msg_data = envelope.get('dataMessage', {})
            text = msg_data.get('message', '')
            group_info = msg_data.get('groupInfo', {})
            group_id = group_info.get('groupId', None)

            if not text:
                sync = envelope.get('syncMessage', {})
                sent = sync.get('sentMessage', {})
                text = sent.get('message', '')
                if not group_id:
                    sync_group = sent.get('groupInfo', {})
                    group_id = sync_group.get('groupId', None)

            return {
                "source": source,
                "text": text,
                "group_id": group_id,
                #"raw": data
            }

        return None

    async def send_article(self, articles, recipient_data):
        if recipient_data.get("group_id"):
            recipient_params = {"groupId": recipient_data["group_id"]}
        else:
            recipient_params = {"recipient": [recipient_data["source"]]}

        for i, article in enumerate(articles):
            image_path = load_image(article["image_url"])
            print(f"DEBUG image_path: {image_path}")
            send = {"jsonrpc": "2.0", "method": "send", "params": {
                **recipient_params,
                "message": article["url"],
                "previewUrl": article["url"],
                "previewTitle": article["title"],
                "previewDescription": article["description"],
                "previewImage": image_path,
            }, "id": 2}

            self.writer.write(json.dumps(send).encode() + b'\n')

            await self.writer.drain()
            asyncio.sleep(2)

    async def send_text(self, message, recipient_data):
        if recipient_data.get("group_id"):
            recipient_params = {"groupId": recipient_data["group_id"]}
        else:
            recipient_params = {"recipient": [recipient_data["source"]]}
        send = {"jsonrpc": "2.0", "method": "send", "params": {
            **recipient_params,
            "message": message,
        }, "id": 2}
        self.writer.write(json.dumps(send).encode() + b'\n')
        await self.writer.drain()
        await asyncio.sleep(2)  # BUG FIX: missing await