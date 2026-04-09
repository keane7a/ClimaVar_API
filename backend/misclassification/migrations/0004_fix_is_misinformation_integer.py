from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('misclassification', '0003_alter_misclassificationlog_is_misinformation'),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='misclassification_misclassificationlog' 
                        AND column_name='is_misinformation'
                        AND data_type='boolean'
                    ) THEN
                        ALTER TABLE misclassification_misclassificationlog 
                        ALTER COLUMN is_misinformation TYPE integer 
                        USING CASE 
                            WHEN is_misinformation = true THEN 1 
                            WHEN is_misinformation = false THEN 0 
                            ELSE 0 
                        END;
                    END IF;
                END $$;
            """,
            reverse_sql=migrations.RunSQL.noop,
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
