from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
import time
from app.config import settings


class GemmaService:
    def __init__(self):
        if not settings.use_gemma_stub:
            self.processor = AutoProcessor.from_pretrained(
                settings.gemma_model_name  # "google/gemma-4"
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                settings.gemma_model_name  #"google/gemma-4"
            )

    def describe_accident(self, frames):
        if not settings.use_gemma_stub:
            print("Начало обработки кадров Gemma 4")
            prompt = """
            Analyze these traffic accident frames.
    
            Describe:
            - accident type
            - involved vehicles
            - collision severity
            - road situation
            - possible consequences
            
            translate your answer to russian if you can
            """

            inputs = self.processor(
                images=frames,
                text=prompt,
                return_tensors="pt"
            )

            outputs = self.model.generate(**inputs, max_new_tokens=512)

            text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]

            return text
        else:
            time.sleep(settings.gemma_stub_sleep_sec)
            return """На предоставленных кадрах изображена дорожно-транспортная авария, произошедшая в условиях сильного дождя.

**Описание:**

*   **Тип аварии:** Столкновение двух автомобилей.
*   **Участники:**
    1.  Белый седан (слева).
    2.  Желтый/бежевый седан (в центре, кажется, пострадавший).
    3.  Красный седан (справа, сильно поврежденный).
*   **Степень тяжести столкновения:** Столкновение выглядит достаточно серьезным. Красный автомобиль сильно поврежден, и на асфальте видны многочисленные осколки и повреждения, что указывает на значительную энергию удара.
*   **Состояние дороги:** Дорога мокрая, что усугубляет ситуацию, так как снижает сцепление с дорогой и замедляет реакцию водителей.
*   **Дорожная обстановка:** Авария произошла на повороте или перекрестке, рядом с ограждением и прилегающей территорией. На заднем плане видна оживленная дорога с несколькими другими транспортными средствами.
*   **Возможные последствия:**
    *   Травмы для водителей и пассажиров обоих автомобилей.
    *   Повреждение кузовов автомобилей.
    *   Затор на дороге из-за заблокированных транспортных средств.
    *   Необходимость в оказании помощи пострадавшим и, возможно, в привлечении полиции.
"""
