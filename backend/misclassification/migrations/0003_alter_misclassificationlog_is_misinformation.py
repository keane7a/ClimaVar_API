from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('misclassification', '0002_misclassificationlog_references'),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                ALTER TABLE misclassification_misclassificationlog 
                ALTER COLUMN is_misinformation TYPE integer 
                USING CASE 
                    WHEN is_misinformation = true THEN 1 
                    WHEN is_misinformation = false THEN 0 
                    ELSE 0 
                END;
            """,
            reverse_sql="""
                ALTER TABLE misclassification_misclassificationlog 
                ALTER COLUMN is_misinformation TYPE boolean 
                USING CASE 
                    WHEN is_misinformation = 1 THEN true 
                    ELSE false 
                END;
            """,
        ),
        migrations.AlterField(
            model_name='misclassificationlog',
            name='is_misinformation',
            field=models.IntegerField(
                null=False,
                default=0,
                choices=[
                    (0, 'Accurate'),
                    (1, 'Misinformation'),
                    (2, 'Partial / Needs context'),
                ],
            ),
        ),
    ]
