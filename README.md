# Descripción
Este script genera un dataset sintético y realista de perfiles profesionales del ámbito de datos (Data Scientist, Data Engineer, Analyst, ML Engineer, etc.) con información coherente entre experiencia, edad y salario, además de campos operativos como del formulario de la página web que lo genera:

created_at

updated_at

consent_accepted

consent_ts


El objetivo es proporcionar un dataset totalmente seguro, anonimizado y plausible para pruebas dentro de un proyecto en AWS (https://github.com/jesus-jpeg/data-salaries-uoc) que incluye:

Carga en RDS
Transofrmaciones en Lambda
Carga en S3
Consulta en Amazon Athena


Este dataset no utiliza ningún dato personal real ya que todos han sido generados de forma automática y random.
